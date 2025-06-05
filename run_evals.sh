#!/bin/bash

datasets=('mars' 'itm' 'coursera')
topks=(10 50)
cv_splits=(5 10)

for dataset in "${datasets[@]}"; do
  for topk in "${topks[@]}"; do
    for cv_split in "${cv_splits[@]}"; do
      echo "Evaluating $dataset dataset on the Top-$topk recommendations with $cv_split splits"
      uv run src/main.py -e -ds "$dataset" --top_k "$topk" -k "$cv_split"
      uv run src/main.py -s -ds "$dataset" --top_k "$topk" -k "$cv_split"
    done
  done
done
