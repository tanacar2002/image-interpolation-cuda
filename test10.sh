#!/bin/sh
max=10
sum=0.000000
minmaxsum=0.000000
subscalesum=0.000000

line_count(){
    out="$($@)"
    time=$(echo "$out" | grep -o " [0-9]*\.[0-9]* ms" | cut -d" " -f 2)
    sum=$(echo "$sum + $time" | bc -l)
}

$@ > /dev/null

for i in `seq 1 $max`
do
    line_count $@
done

echo "Average execution time: $(echo "$sum/$max" | bc -l) ms"