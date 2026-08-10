# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for constructing raysim meshes from NumPy arrays."""

import numpy as np
import pytest
import raysim.cuda as rs


def octahedron_geometry():
    """Return a closed octahedron centred in front of the probe."""
    vertices = np.array(
        [
            [0.0, 0.0, -60.0],
            [20.0, 0.0, -80.0],
            [0.0, 20.0, -80.0],
            [-20.0, 0.0, -80.0],
            [0.0, -20.0, -80.0],
            [0.0, 0.0, -100.0],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [5, 2, 1],
            [5, 3, 2],
            [5, 4, 3],
            [5, 1, 4],
        ],
        dtype=np.uint32,
    )
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    return vertices, indices, normals


@pytest.mark.parametrize("index_dtype", [np.uint32, np.int32, np.int64])
def test_array_constructor_accepts_supported_index_dtypes(index_dtype):
    vertices, indices, normals = octahedron_geometry()

    mesh = rs.Mesh(
        vertices=vertices,
        indices=indices.astype(index_dtype),
        normals=normals,
        material_id=0,
    )

    np.testing.assert_array_equal(mesh.get_aabb_min(), [-20.0, -20.0, -100.0])
    np.testing.assert_array_equal(mesh.get_aabb_max(), [20.0, 20.0, -60.0])


def test_array_constructor_copies_non_contiguous_inputs():
    vertices, indices, normals = octahedron_geometry()
    vertices = np.asfortranarray(vertices)
    indices = np.asfortranarray(indices)
    normals = np.asfortranarray(normals)

    mesh = rs.Mesh(
        vertices=vertices,
        indices=indices,
        normals=normals,
        material_id=0,
    )

    np.testing.assert_array_equal(mesh.get_aabb_min(), [-20.0, -20.0, -100.0])
    np.testing.assert_array_equal(mesh.get_aabb_max(), [20.0, 20.0, -60.0])


def unaligned_copy(array):
    """Copy an array to a C-contiguous buffer with an unaligned data pointer."""
    storage = np.empty(array.nbytes + 1, dtype=np.uint8)
    result = np.ndarray(array.shape, dtype=array.dtype, buffer=storage, offset=1)
    result[...] = array
    assert result.flags.c_contiguous
    assert not result.flags.aligned
    return result


def test_array_constructor_copies_unaligned_inputs():
    vertices, indices, normals = octahedron_geometry()

    mesh = rs.Mesh(
        vertices=unaligned_copy(vertices),
        indices=unaligned_copy(indices),
        normals=unaligned_copy(normals),
        material_id=0,
    )

    np.testing.assert_array_equal(mesh.get_aabb_min(), [-20.0, -20.0, -100.0])
    np.testing.assert_array_equal(mesh.get_aabb_max(), [20.0, 20.0, -60.0])


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("vertices", np.zeros((3, 2), dtype=np.float32), r"vertices must have shape \(N, 3\)"),
        ("vertices", np.zeros((3, 3), dtype=np.float64), "vertices must have dtype float32"),
        ("indices", np.zeros((3, 2), dtype=np.uint32), r"indices must have shape \(M, 3\)"),
        (
            "indices",
            np.zeros((3, 3), dtype=np.float32),
            "indices must have dtype uint32, int32, or int64",
        ),
        ("normals", np.zeros((3, 2), dtype=np.float32), r"normals must have shape \(N, 3\)"),
        ("normals", np.zeros((6, 3), dtype=np.float64), "normals must have dtype float32"),
        (
            "normals",
            np.zeros((5, 3), dtype=np.float32),
            "normals must have the same number of rows as vertices",
        ),
    ],
)
def test_array_constructor_rejects_shape_and_dtype_errors(argument, value, message):
    vertices, indices, normals = octahedron_geometry()
    arguments = {
        "vertices": vertices,
        "indices": indices,
        "normals": normals,
        "material_id": 0,
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        rs.Mesh(**arguments)


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (
            np.array([[0, -1, 2]], dtype=np.int32),
            "indices must contain only non-negative values",
        ),
        (
            np.array([[0, 1, np.iinfo(np.uint32).max + 1]], dtype=np.int64),
            "indices values must fit in uint32",
        ),
        (
            np.array([[0, 1, 6]], dtype=np.uint32),
            "Mesh: index is out of range for vertices",
        ),
    ],
)
def test_array_constructor_checks_index_bounds(indices, message):
    vertices, _, normals = octahedron_geometry()

    with pytest.raises(ValueError, match=message):
        rs.Mesh(vertices=vertices, indices=indices, normals=normals, material_id=0)


@pytest.mark.parametrize(
    ("vertices", "indices", "message"),
    [
        (
            np.empty((0, 3), dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.uint32),
            "Mesh: vertices must not be empty",
        ),
        (
            np.zeros((3, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint32),
            "Mesh: indices must not be empty",
        ),
    ],
)
def test_array_constructor_rejects_empty_geometry(vertices, indices, message):
    with pytest.raises(ValueError, match=message):
        rs.Mesh(vertices=vertices, indices=indices, material_id=0)


def write_obj(file_name, vertices, indices, normals):
    """Write geometry without changing its vertex or normal indexing."""
    lines = ["o Octahedron"]
    lines.extend("v " + " ".join(f"{float(value):.9g}" for value in vertex) for vertex in vertices)
    lines.extend("vn " + " ".join(f"{float(value):.9g}" for value in normal) for normal in normals)
    lines.extend(
        "f " + " ".join(f"{int(index) + 1}//{int(index) + 1}" for index in triangle)
        for triangle in indices
    )
    file_name.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_mesh(mesh_factory):
    """Render one small deterministic frame for a supplied mesh factory."""
    materials = rs.Materials()
    material_id = materials.get_index("fat")
    world = rs.World("water")
    world.add(mesh_factory(material_id))

    simulator = rs.RaytracingUltrasoundSimulator(world, materials)
    probe = rs.LinearArrayProbe(
        pose=rs.Pose(
            position=np.zeros(3, dtype=np.float32),
            rotation=np.array([0.0, np.pi, 0.0], dtype=np.float32),
        ),
        num_elements_x=32,
        width=40.0,
        num_el_samples=1,
    )
    sim_params = rs.SimParams()
    sim_params.t_far = 120.0
    sim_params.buffer_size = 4096
    sim_params.max_depth = 3
    sim_params.use_scattering = False
    sim_params.conv_psf = False
    sim_params.b_mode_size = (64, 64)
    return simulator.simulate(probe, sim_params)


@pytest.mark.gpu
def test_array_and_file_meshes_render_identically(tmp_path):
    vertices, indices, normals = octahedron_geometry()
    obj_file = tmp_path / "octahedron.obj"
    write_obj(obj_file, vertices, indices, normals)

    array_image = render_mesh(
        lambda material_id: rs.Mesh(
            vertices=vertices,
            indices=indices,
            normals=normals,
            material_id=material_id,
        )
    )
    file_image = render_mesh(lambda material_id: rs.Mesh(str(obj_file), material_id))

    np.testing.assert_allclose(array_image, file_image, rtol=0.0, atol=1e-6)


@pytest.mark.gpu
def test_omitted_normals_produce_a_renderable_mesh():
    vertices, indices, _ = octahedron_geometry()

    image = render_mesh(
        lambda material_id: rs.Mesh(
            vertices=vertices,
            indices=indices,
            material_id=material_id,
        )
    )

    assert image.shape == (64, 64)
