#!/bin/bash
LINE="["

# for i in "./build/cpu/cpu" "./build/cpu_cv/cpu_cv" "./build/gpu/gpu" "./build/gpu_texture/gpu_texture" "./build/gpu_cv/gpu_cv"
# do
# LINE+="["
for j in "./samples/1.jpg" "./samples/2.jpg"
do 
LINE+="["
for f in 2 4
do
LINE+="["
for alg in "nn_v2" "lin" "cub_v2" "lan_v3" # "nn" "lin" "cub" "lan" "nn_a" "lin_a" "cub_a" "lan_a" #
do
    LINE+="$(./test10.sh $1 $j $f $f $alg | grep -o " [0-9]*\.[0-9]* ms" | cut -d" " -f 2), "
    echo "$1 $j $alg"
done
LINE+="], "
done
LINE+="], "
done
LINE+="], "
# done
# LINE+="]"

echo "$LINE"