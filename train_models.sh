#!/bin/bash

echo "-----TRAINING MPNN MODEL WITHOUT DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "mpnn_no_distance_concat" --model "MPNN" --load_params "mpnn_config.json" --epochs 100 --gpu > mpnn_no_distance_concat.out


echo "-----TRAINING MPNN MODEL WITH DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "mpnn_distance_concat" --model "MPNN" --load_params "mpnn_config.json" --epochs 100 --gpu --include_distance > mpnn_distance_concat.out


echo "-----TRAINING GCN MODEL WITHOUT DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gcn_no_distance_concat" --model "GCN" --load_params "gcn_config.json" --epochs 100 --gpu > gcn_no_distance_concat.out


echo "-----TRAINING GCN MODEL WITH DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gcn_distance_concat" --model "GCN" --load_params "gcn_config.json" --epochs 100 --gpu --include_distance > gcn_distance_concat.out


echo "-----TRAINING GFCN MODEL WITHOUT DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gfcn_no_distance_concat" --model "GFCN" --load_params "gfcn_config.json" --epochs 100 --gpu > gfcn_no_distance_concat.out


echo "-----TRAINING GFCN MODEL WITH DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gfcn_distance_concat" --model "GFCN" --load_params "gfcn_config.json" --epochs 100 --gpu --include_distance > gfcn_distance_concat.out


echo "-----TRAINING GAT MODEL WITHOUT DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gat_no_distance_concat" --model "GAT" --load_params "gat_config.json" --epochs 100 --gpu > gat_no_distance_concat.out


echo "-----TRAINING GAT MODEL WITH DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gat_distance_concat" --model "GAT" --load_params "gat_config.json" --epochs 100 --gpu --include_distance > gat_distance_concat.out

