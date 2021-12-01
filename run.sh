#!/bin/bash

echo "Target file: $1"
echo "Model : $2"
cmd="python src/main.py --src $1 --model $2"
docker run --rm -it -v $PWD/data:/opt/data -v $PWD/models:/opt/models -v $PWD/outputs:/opt/outputs tflite-ht $cmd