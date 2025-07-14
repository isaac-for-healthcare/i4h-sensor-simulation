# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
