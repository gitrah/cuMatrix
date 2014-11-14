#!/bin/bash
#
grep -R $@  | grep -oE '.*c[cu]:' | grep -oP '.*[^:]' | xargs touch
