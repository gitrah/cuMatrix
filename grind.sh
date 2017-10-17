# !/bin/sh
#
valgrind --tool=callgrind ./cumatrel
./gprof2dot.py -f callgrind callgrind.out.x | dot -Tsvg -o output.svg
