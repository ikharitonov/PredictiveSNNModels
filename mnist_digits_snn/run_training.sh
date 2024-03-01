#!/bin/bash

echo "SCRIPT STARTED:"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

source /home/ikharitonov/anaconda3/bin/activate
conda activate snntorch

python snntorch_training.py

conda deactivate

echo "SCRIPT FINISHED:"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
