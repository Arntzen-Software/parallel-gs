#!/bin/bash

# This is run from inside the container.
# Useful as an initial template.

mkdir -p build-steamrt
cd build-steamrt
cmake .. \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=$(pwd)/../output-steamrt \
	-DPYTHON_EXECUTABLE=$(which python3) \
	-G Ninja

ninja install/strip -v

