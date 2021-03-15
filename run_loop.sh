#! /bin/bash

for ((i=0;i<=10;i++))
do
	echo $i
	./run_cifar.sh
done

