ls $1/*.o | xargs nm  -C -A | grep $2
