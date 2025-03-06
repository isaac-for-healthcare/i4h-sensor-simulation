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

include(CMakeParseArguments)

# Generate a C++ header file from a text file.
#
# The header file will contain a char array and will be generated in
#
#  ${CMAKE_CURRENT_BINARY_DIR}\DIR\ARRAY_NAME.hpp
#
# The array name is build from FILE_PATH by using the file name and replacing `-`, and `.` by `_`.
# Example usage in CMakeLists.txt:
#
#   add_library(fonts_target SHARED)
#   gen_header_from_text_file(TARGET font_target FILE_PATH `text\text.txt`)
#
# This will generate the `text\text.hpp` in the current binary dir. The file contains an
# array named `text`. The file is added to the project and the current binary dir is added
# to the include paths.
# To use the generated header in the code:
#
#   #include <text\text.hpp>
#   void func() {
#       cout << text << std::endl;
#   }
#
# Usage:
#
#     gen_header_from_text_file (FILE_PATH <PATH> [TARGET <TGT>])
#
#   ``FILE_PATH``
#     text file to convert to a header relative to CMAKE_CURRENT_SOURCE_DIR
#   ``TARGET``
#     if specified the generated header file and the include path is added to the target

function(gen_header_from_text_file)
    set(_options)
    set(_singleargs FILE_PATH TARGET)
    set(_multiargs)

    cmake_parse_arguments(gen_header "${_options}" "${_singleargs}" "${_multiargs}" "${ARGN}")

    # build the array name
    string(REPLACE "${CMAKE_CURRENT_BINARY_DIR}/" "" RELATIVE_FILE_PATH ${gen_header_FILE_PATH})
    get_filename_component(FILE_ARRAY_NAME ${RELATIVE_FILE_PATH} NAME)
    string(TOLOWER ${FILE_ARRAY_NAME} FILE_ARRAY_NAME)
    string(REPLACE "-" "_" FILE_ARRAY_NAME ${FILE_ARRAY_NAME})
    string(REPLACE "." "_" FILE_ARRAY_NAME ${FILE_ARRAY_NAME})

    # create the header file
    get_filename_component(FILE_DIRECTORY ${RELATIVE_FILE_PATH} DIRECTORY)
    set(HEADER_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/${FILE_DIRECTORY}/${FILE_ARRAY_NAME}.hpp")

    # Use the ADD_STRING_LITERAL_SCRIPT if defined, otherwise use the script in the same directory
    if(DEFINED ADD_STRING_LITERAL_SCRIPT)
        set(SCRIPT_PATH ${ADD_STRING_LITERAL_SCRIPT})
    else()
        set(SCRIPT_PATH ${CMAKE_CURRENT_LIST_DIR}/add_string_literal.sh)
    endif()

    message(STATUS "Using add_string_literal script: ${SCRIPT_PATH}")

    add_custom_command(
        OUTPUT ${HEADER_FILE_NAME}
        COMMAND ${SCRIPT_PATH} ${FILE_ARRAY_NAME} ${HEADER_FILE_NAME} ${gen_header_FILE_PATH}
        COMMENT "Created header ${HEADER_FILE_NAME} from text file '${gen_header_FILE_PATH}'"
        DEPENDS ${gen_header_FILE_PATH} ${SCRIPT_PATH}
        )

    if(gen_header_TARGET)
        # add the created file to the target
        target_sources(${gen_header_TARGET}
            PRIVATE
                ${HEADER_FILE_NAME}
            )
        # also add the binary dir to include directories so the header can be found
        target_include_directories(${gen_header_TARGET}
            PRIVATE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/${FILE_DIRECTORY}
            )
    endif()

endfunction()
