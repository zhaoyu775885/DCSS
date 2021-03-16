#! /bin/bash

n_loop=5
for ((i=1;i<=n_loop;i++))
do
	echo $i-th "run"
	./run_cifar.sh
done

