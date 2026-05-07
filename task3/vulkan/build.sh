#!/usr/bin/env bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

glslc -fshader-stage=comp "$parent_path"/shaders/histogram.comp       -o "$parent_path"/spv/histogram.spv
glslc -fshader-stage=comp "$parent_path"/shaders/find_threshold.comp  -o "$parent_path"/spv/find_threshold.spv
glslc -fshader-stage=comp "$parent_path"/shaders/apply_threshold.comp -o "$parent_path"/spv/apply_threshold.spv

cd "$parent_path"
cmake .
make 