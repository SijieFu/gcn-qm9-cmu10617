#!/bin/bash

echo "-----TRAINING MPNN MODEL WITHOUT DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "mpnn_no_distance" --model "MPNN" --load_params "mpnn_config.json" --epochs 100 --gpu --concat_hidden > mpnn_no_distance.out


echo "-----TRAINING MPNN MODEL WITH DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "mpnn_distance" --model "MPNN" --load_params "mpnn_config.json" --epochs 100 --gpu --include_distance --concat_hidden > mpnn_distance.out


echo "-----TRAINING GCN MODEL WITHOUT DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gcn_no_distance" --model "GCN" --load_params "gcn_config.json" --epochs 100 --gpu --concat_hidden > gcn_no_distance.out


echo "-----TRAINING GCN MODEL WITH DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gcn_distance" --model "GCN" --load_params "gcn_config.json" --epochs 100 --gpu --include_distance --concat_hidden > gcn_distance.out


echo "-----TRAINING GFCN MODEL WITHOUT DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gfcn_no_distance" --model "GFCN" --load_params "gfcn_config.json" --epochs 100 --gpu --concat_hidden > gfcn_no_distance.out


echo "-----TRAINING GFCN MODEL WITH DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gfcn_distance" --model "GFCN" --load_params "gfcn_config.json" --epochs 100 --gpu --include_distance --concat_hidden > gfcn_distance.out


echo "-----TRAINING GAT MODEL WITHOUT DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gat_no_distance" --model "GAT" --load_params "gat_config.json" --epochs 100 --gpu --concat_hidden > gat_no_distance.out


echo "-----TRAINING GAT MODEL WITH DISTANCE-----"
python main.py --model_path "./final_models/" --out_file "gat_distance" --model "GAT" --load_params "gat_config.json" --epochs 100 --gpu --include_distance --concat_hidden > gat_distance.out

