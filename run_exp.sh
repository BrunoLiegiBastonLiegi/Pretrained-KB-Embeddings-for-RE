#!/bin/bash

dir=$1
train=$2
test=$3

file="${dir}results_kg.json"

echo { > $file

for i in {1..10}
do
    echo "Beginning run $i..."
    echo \"run_$i\": >> $file
    python run_experiment.py $train $test
    if [ $i -ne 10 ]
    then
	echo , >> $file
    fi
done
echo } >> $file
