#!/bin/bash

# Useful as an initial template.

docker run -it --init --rm -u $UID -w /granite -v $(pwd):/granite registry.gitlab.steamos.cloud/steamrt/sniper/sdk ./scripts/build-steamrt-inside.sh

