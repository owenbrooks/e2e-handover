#!/bin/bash
command="mv image_rgb_1/ image_rgb_1_orig"
find . -maxdepth 1 -type d \( ! -name . \) -exec bash -c "cd '{}' && $command" \;

