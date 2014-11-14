#!/bin/sh
# pretty-prints undefined references output to stderr by g-/nvcc
#                                        
# Usage: '<<make expression of choice>> |& undefdRefs.sh
#
# (of course, first bildDmg.sh to build the demangler 'dmg')
#
grep 'Undefined reference to' | awk -F"'" '{print $2}' | xargs dmg
