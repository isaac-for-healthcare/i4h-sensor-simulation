import os
import numpy as np
import scipy
from typing import Tuple

from holoscan.core import (
    Application,
    Operator,
    OperatorSpec,
    MetadataPolicy,
    Tracker
)

from scipy.interpolate import griddata

from holoscan.operators import HolovizOp


def scan_convert_curvilinear(scan_lines, angles, depths, radius=5, x_size=500, z_size=500):
    """
    Convert curvilinear scan data to Cartesian coordinates for display.

    Parameters:
    -----------
    scan_lines : ndarray
        2D array of shape [n_angles, n_depths].
    angles : ndarray
        Array of angles in radians for each scan line [rad].
    depths : ndarray
        Array of depths for each sample along scan lines.
    radius : float
        Radius of curvature of the transducer in mm.
    x_size : int
        Width of output image in pixels.
    z_size : int
        Height of output image in pixels.

    Returns:
    --------
    grid_z : ndarray
        2D Cartesian scan-converted image.
    x_cart, z_cart : ndarray
        Axes for the grid_z image (mm).
    """
    angles = angles - np.pi / 2
    angle_mesh, depth_mesh = np.meshgrid(angles, depths)

    # Convert polar to Cartesian coordinates with curved origin
    z_mesh = (radius + depth_mesh) * np.sin(angle_mesh) + radius
    x_mesh = (radius + depth_mesh) * np.cos(angle_mesh)

    # Create a regular grid for interpolation
    x_max = np.max(x_mesh)
    z_max = np.max(z_mesh)
    x_min = np.min(x_mesh)
    z_min = np.min(z_mesh)
    x_cart = np.linspace(x_min, x_max, x_size)
    z_cart = np.linspace(z_min, z_max, z_size)
    xi, zi = np.meshgrid(x_cart, z_cart)

    # Interpolate
    points = np.column_stack((x_mesh.flatten(), z_mesh.flatten()))
    values = scan_lines.flatten()
    grid_z = np.asarray(griddata(points, values, (xi, zi), method='cubic', fill_value=np.nan))

    # Mask transducer area (remove data inside the probe radius)
    grid_z[np.sqrt(xi**2 + (zi - radius)**2) < radius] = np.nan

    return grid_z, x_cart, z_cart


def create_piece_wise_tgc(depth_samples: int,
                          control_points: Tuple[Tuple[float, float], ...],
                          c: float = 1540,
                          fs: float = 50e6):
    """
    Create a piece-wise linear TGC curve from control points.

    Parameters:
    -----------
    depth_samples : int
        Number of samples along depth.
    control_points : list of tuples
        List of (depth_cm, gain_db) pairs.
    c : float, optional
        Speed of sound in tissue (m/s).
    fs : float, optional
        Sampling frequency (Hz).

    Returns:
    --------
    tgc_curve : ndarray
        1D array of length depth_samples containing the TGC multiplier.
    """
    # Calculate time vector
    t = np.arange(depth_samples) / fs
    # Calculate depth in cm (two-way travel)
    depth = (c * t / 2) * 100.0  # in cm

    control_depths, control_gains = zip(*control_points)
    tgc_curve_db = np.interp(depth, np.asarray(control_depths), np.asarray(control_gains))
    tgc_curve = 10 ** (tgc_curve_db / 20)

    # Normalize so TGC starts at 1
    tgc_curve = tgc_curve / tgc_curve[0]
    return tgc_curve


class DataGenerator(Operator):
    """
    Data generator operator that reads RF data from 'demo_scanlines.npy' and emits it with metadata.
    """

    def __init__(self,
                 fragment,
                 *args,
                 file_path: str = None,
                 axis: int = -1,
                 scan_depth: float = None,
                 num_elements: int = None,
                 opening_angle: float = None,
                 **kwargs):
        self.file_path = file_path or os.path.join(os.path.dirname(__file__), 'demo_scanlines.npy')
        self.axis = axis
        self.scan_depth = scan_depth
        self.num_elements = num_elements
        self.opening_angle = opening_angle
        self.data = None
        self.buffer_size = None
        super().__init__(fragment, *args, **kwargs)


    def setup(self, spec: OperatorSpec):
        spec.output("output")

        if (self.scan_depth is None or
            self.num_elements is None or
            self.opening_angle is None):
            raise ValueError("Missing parameter(s): scan_depth, num_elements, and opening_angle are required.")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found")

        # Load the data
        self.data = np.load(self.file_path)

        if self.axis < 0:
            self.axis = self.data.ndim + self.axis

        # The depth is the size along the specified axis
        self.buffer_size = self.data.shape[self.axis]

    def compute(self, op_input, op_output, context):
        self.metadata["axis"] = self.axis
        self.metadata["depth"] = self.buffer_size
        self.metadata["depth_axis"] = np.linspace(0, self.scan_depth, self.buffer_size)
        self.metadata["angle_axis"] = np.linspace(-self.opening_angle / 2,
                                                  self.opening_angle / 2,
                                                  self.num_elements)

        # Emit data + metadata
        op_output.emit(self.data, "output")


class TimeGainCompensation(Operator):
    """
    Applies piecewise linear TGC to the RF data.
    """

    def __init__(self,
                 fragment,
                 *args,
                 control_points: Tuple[Tuple[float, float], ...] = ((0, 0), (4, 4)),
                 **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.control_points = control_points
        self.tgc_func = None

    def setup(self, spec: OperatorSpec):
        spec.input("rf_input")
        spec.output("rf_output")

    def compute(self, op_input, op_output, context):
        rf_signal = op_input.receive("rf_input")
        axis = self.metadata["axis"]
        signal_depth = self.metadata["depth"]

        # Create TGC the first time we run
        if self.tgc_func is None:
            self.tgc_func = create_piece_wise_tgc(signal_depth, self.control_points)

        if rf_signal.shape[axis] != signal_depth:
            raise ValueError(f"Input shape along axis {axis} != {signal_depth}")

        shape = [1] * rf_signal.ndim
        shape[axis] = signal_depth
        tgc_extended = self.tgc_func.reshape(shape)

        compensated = rf_signal * tgc_extended
        op_output.emit(compensated, "rf_output")


class EnvelopeDetection(Operator):
    """
    Extracts amplitude envelope using the Hilbert transform.
    """

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        rf_signal = op_input.receive("input")
        axis = self.metadata["axis"]

        analytic_signal = scipy.signal.hilbert(rf_signal, axis=axis)
        envelope = np.abs(np.asarray(analytic_signal))

        op_output.emit(envelope, "output")


class LogCompression(Operator):
    """
    Log compress the envelope data.
    """

    def __init__(self, fragment, *args, max_quantile=0.99, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.max_quantile = max_quantile
        self.max_value = None

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        envelope = op_input.receive("input")

        # We'll just do a global percent quantile for the entire dataset.
        if self.max_value is None:
            self.max_value = np.quantile(envelope, self.max_quantile)

        normalized = envelope / self.max_value
        # Guard against log(0)
        normalized[normalized <= 0] = np.finfo(normalized.dtype).eps
        log_compressed = 20 * np.log10(normalized)

        op_output.emit(log_compressed, "output")


class CurvilinearScanConversion(Operator):
    """
    Convert polar data into Cartesian coordinates for final B-mode image.
    """

    def __init__(self,
                 fragment,
                 *args,
                 probe_radius=10.0,
                 x_size=500,
                 z_size=500,
                 **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.probe_radius = probe_radius
        self.x_size = x_size
        self.z_size = z_size

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        polar_signal = op_input.receive("input")

        depth_axis = self.metadata["depth_axis"]
        angle_axis = self.metadata["angle_axis"]

        bmode, x_cart, z_cart = scan_convert_curvilinear(
            polar_signal,
            angle_axis,
            depth_axis,
            radius=self.probe_radius,
            x_size=self.x_size,
            z_size=self.z_size
        )

        # Attach Cartesian axes to metadata so the next operator (visualizer) can use them
        self.metadata["x_cart"] = x_cart
        self.metadata["z_cart"] = z_cart

        bmode = np.flipud(bmode)
        mn = -40  # dB
        mx = 0  # dB

        mx -= mn

        bmode = ((bmode - mn)/mx) * 255
        bmode = np.nan_to_num(bmode, nan=255)
        bmode = np.clip(bmode, 0, 255)
        bmode = bmode.astype(np.uint8)

        rgba_data = np.stack(
            [
                bmode,         # R
                bmode,         # G
                bmode,         # B
                np.full_like(bmode, 255, dtype=np.uint8),  # A
            ],
            axis=-1
        )



        # Emit the RGBA matrix
        op_output.emit({"": rgba_data}, "output")



class SimulationPostProcessing(Application):
    def compose(self):
        data_gen = DataGenerator(
            self,
            name="data_gen",
            file_path=os.path.join(os.path.dirname(__file__), "demo_scanlines.npy"),
            scan_depth=40.0,     # mm
            num_elements=128,
            opening_angle=0.523599 * 2,  # ~60 degrees in radians
            axis=0
        )
        tcg = TimeGainCompensation(self, name="tgc", control_points=((0, 0), (4, 4)))
        env_det = EnvelopeDetection(self, name="env_det")
        lc = LogCompression(self, name="lc", max_quantile=0.99)
        sc = CurvilinearScanConversion(self, name="sc", probe_radius=10.0)
        holoviz_op = HolovizOp(
            self,
            name="holoviz",
            width=512,
            height=512,
            tensors=[
                # `name=""` here to match the output of VideoStreamReplayerOp
                dict(name="", type="color", opacity=1.0, priority=0),
            ],

        )

        # Optionally specify metadata policies
        tcg.metadata_policy = MetadataPolicy.RAISE
        lc.metadata_policy = MetadataPolicy.RAISE
        sc.metadata_policy = MetadataPolicy.RAISE
        holoviz_op.metadata_policy = MetadataPolicy.RAISE

        # Connect flows
        self.add_flow(data_gen, tcg)
        self.add_flow(tcg, env_det)
        self.add_flow(env_det, lc)
        self.add_flow(lc, sc)
        # Finally connect the scan conversion output to the visualizer
        self.add_flow(sc, holoviz_op, {("output", "receivers")})


def main(config_file=None):
    app = SimulationPostProcessing()
    with Tracker(app, filename="tracker_logger.log", num_start_messages_to_skip=2, num_last_messages_to_discard=3) as tracker:
        # Enable metadata before running
        app.is_metadata_enabled = True
        app.run()



if __name__ == "__main__":
    main()
