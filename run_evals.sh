#!/bin/bash

datasets=('mars' 'itm' 'coursera')

for dataset in "${datasets[@]}"; do
  echo "Evaluating $dataset dataset"
  # uv run src/main.py -e -ds "$dataset"
  uv run src/main.py -s -ds "$dataset"
done
