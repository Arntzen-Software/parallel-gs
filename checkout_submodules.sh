#!/bin/bash

# Only checks out what is necessary to build standalone.

update() {
	git submodule sync $1
	git submodule update --init $1
}

update Granite
cd Granite
update third_party/volk
update third_party/khronos/vulkan-headers
