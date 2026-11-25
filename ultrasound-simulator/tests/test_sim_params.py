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

"""Tests for the SimParams Python API."""

import numpy as np
import pytest
import ultrasound_simulator.cuda as rs

NUMPY_DTYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
]


@pytest.mark.parametrize("dtype", NUMPY_DTYPES, ids=lambda d: np.dtype(d).name)
def test_b_mode_size_reads_back_numpy_arrays_of_every_dtype(dtype):
    params = rs.SimParams()
    params.b_mode_size = np.array([120, 100], dtype=dtype)
    assert tuple(params.b_mode_size) == (120, 100)


def test_b_mode_size_reads_back_numpy_default_arrays():
    params = rs.SimParams()
    params.b_mode_size = np.array([500, 400])
    assert tuple(params.b_mode_size) == (500, 400)
    params.b_mode_size = np.array([500.0, 400.0])
    assert tuple(params.b_mode_size) == (500, 400)


def test_b_mode_size_reads_back_a_strided_numpy_view():
    params = rs.SimParams()
    params.b_mode_size = np.array([500, 0, 400, 0], dtype=np.int64)[::2]
    assert tuple(params.b_mode_size) == (500, 400)
    params.b_mode_size = np.array([500, 400], dtype=">i4")  # non-native byte order
    assert tuple(params.b_mode_size) == (500, 400)


@pytest.mark.parametrize("size", [(500, 400), [500, 400]], ids=["tuple", "list"])
def test_b_mode_size_reads_back_tuples_and_lists(size):
    params = rs.SimParams()
    params.b_mode_size = size
    assert tuple(params.b_mode_size) == (500, 400)


@pytest.mark.parametrize(
    "values",
    [
        np.array([-1, 400], dtype=np.int64),
        np.array([500.5, 400.0]),
        np.array([np.nan, 400.0]),
        np.array([np.inf, 400.0]),
        np.array([2**32, 400], dtype=np.int64),
    ],
    ids=["negative", "fraction", "nan", "inf", "above-uint32"],
)
def test_b_mode_size_refuses_numpy_values_a_size_cannot_hold(values):
    params = rs.SimParams()
    params.b_mode_size = (500, 400)
    with pytest.raises(RuntimeError, match="whole numbers"):
        params.b_mode_size = values
    assert tuple(params.b_mode_size) == (500, 400)


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps,
    reason="np.longdouble is no wider than float64 here, so the fraction cannot be written",
)
def test_b_mode_size_refuses_a_longdouble_fraction_that_rounds_to_a_whole_double():
    params = rs.SimParams()
    params.b_mode_size = (500, 400)
    values = np.array([np.longdouble(500) + np.longdouble(2) ** -50, 400], dtype=np.longdouble)
    assert float(values[0]) == 500.0  # the fraction is lost in a double
    with pytest.raises(RuntimeError, match="whole numbers"):
        params.b_mode_size = values
    assert tuple(params.b_mode_size) == (500, 400)


@pytest.mark.parametrize(
    "values",
    [np.array([True, False]), np.array([500 + 0j, 400 + 0j])],
    ids=["bool", "complex"],
)
def test_b_mode_size_refuses_non_numeric_numpy_dtypes(values):
    params = rs.SimParams()
    with pytest.raises(RuntimeError, match="Unsupported numpy array data type"):
        params.b_mode_size = values


def test_b_mode_size_refuses_numpy_arrays_of_the_wrong_shape():
    params = rs.SimParams()
    with pytest.raises(RuntimeError, match=r"shape \(2,\)"):
        params.b_mode_size = np.array([500, 400, 300])
