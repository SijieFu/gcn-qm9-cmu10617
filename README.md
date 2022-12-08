# gcn-qm9-cmu10617 : [final models here](https://drive.google.com/drive/folders/1Vfkx__LlilUYKSYfGGRH7IdNWCK18_SK?usp=share_link)
Final team project repo for CMU 10-617 (Fall 2022)
TorchDrug is forked from:[here](https://github.com/DeepGraphLearning/torchdrug.git) and modifications have been made
# Clone the repo
    $ git clone https://github.com/SijieFu/gcn-qm9-cmu10617.git
# Building the environent
First create and activate the environment: 

    $ conda create -n torchdrug python=3.9; conda activate torchdrug

Next install pytorch (Linux): 

    $ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

Now install pytorch-scatter and pytorch-cluster (Linux): 
    
    $ conda install pytorch-scatter pytorch-cluster -c pyg

Install the requirements: 

    $ pip install -r requirements.txt

# Training a model with `main.py`
##### TAs: `add the --minitest flag to train on 100 samples for a fast check of our code`
##### Training a MPNN with distance `use --gpu if you have GPUs`
    $ python main.py --model_path "./my_models/" --model "MPNN" --out_file "mpnn" --load_params "mpnn_config.json" --epochs 100 --gpu --include_distance
##### Training a GCN with distance `use --gpu if you have GPUs`
    $ python main.py --model_path "./my_models/" --model "GCN" --out_file "gcn" --load_params "gcn_config.json" --epochs 100 --gpu --include_distance
##### Training a GAT with distance `use --gpu if you have GPUs`
    $ python main.py --model_path "./my_models/" --model "GAT" --out_file "gat" --load_params "gat_config.json" --epochs 100 --gpu --include_distance
