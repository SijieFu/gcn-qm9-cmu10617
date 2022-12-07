#!/bin/bash

echo "-----TRAINING MPNN MODEL WITHOUT DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "mpnn_no_distance" --model "MPNN" --load_params "mpnn_config.json" --epochs 100 --gpu > mpnn_no_distance.out


echo "-----TRAINING MPNN MODEL WITH DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "mpnn_distance" --model "MPNN" --load_params "mpnn_config.json" --epochs 100 --gpu --include_distance > mpnn_distance.out


echo "-----TRAINING GCN MODEL WITHOUT DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gcn_no_distance" --model "GCN" --load_params "gcn_config.json" --epochs 100 --gpu > gcn_no_distance.out


echo "-----TRAINING GCN MODEL WITH DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gcn_distance" --model "GCN" --load_params "gcn_config.json" --epochs 100 --gpu --include_distance > gcn_distance.out
