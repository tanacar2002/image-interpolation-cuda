#!/bin/bash
MIN=3
LINE="["

for i in "./samples/640x426.bmp" "./samples/1280x843.bmp" "./samples/1920x1280.bmp" "./samples/5184x3456.bmp"
do
    LINE+="$(./test10.sh $1 $i | grep -o "execution time: [0-9]*\.[0-9]* ms" | cut -d" " -f 3), "
    echo $i
done
LINE+="]"

echo $LINE