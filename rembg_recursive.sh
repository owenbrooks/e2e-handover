#!/bin/bash
command="rembg p image_rgb_1_orig/ image_rgb_1 --model u2netp"
find . -maxdepth 1 -type d \( ! -name . \) -exec bash -c "cd '{}' && $command" \;

