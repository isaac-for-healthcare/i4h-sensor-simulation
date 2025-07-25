/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CPP_PROBE_TYPES
#define CPP_PROBE_TYPES

namespace raysim {

/// Enumeration of supported probe types
enum ProbeType {
  PROBE_TYPE_CURVILINEAR = 0,
  PROBE_TYPE_LINEAR_ARRAY = 1,
  PROBE_TYPE_PHASED_ARRAY = 2
};

}  // namespace raysim

#endif /* CPP_PROBE_TYPES */
