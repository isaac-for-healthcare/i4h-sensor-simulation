import numpy as np
import raysim.cuda as rs


def test_simulation_runs():
    materials = rs.Materials()
    world = rs.World("water")
    material_idx = materials.get_index("fat")
    sphere = rs.Sphere([0, 0, -145], 40, material_idx)
    world.add(sphere)
    simulator = rs.RaytracingUltrasoundSimulator(world, materials)
    probe = rs.CurvilinearProbe(rs.Pose(position=[0, 0, 0], rotation=[0, np.pi, 0]))
    sim_params = rs.SimParams()
    sim_params.t_far = 180.0
    b_mode_image = simulator.simulate(probe, sim_params)
    assert b_mode_image is not None
