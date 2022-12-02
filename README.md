# gcn-qm9-cmu10617
Final team project repo for CMU 10-617 (Fall 2022)
# Builing the environent
First create and activate the environment: 

`conda create -n torchdrug python=3.9; conda activate torchdrug`

Next install pytorch (Linux): 

`conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge`

Next install pytorch (Mac): 

`pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1`

Now install pytorch-scatter and pytorch-cluster: 

`conda install pytorch-scatter -c pyg`

`conda install pytorch-cluster -c pyg`

Clone this repo: 

`git clone https://github.com/SijieFu/gcn-qm9-cmu10617.git; cd gcn-qm9-cmu10617/torchdrug`

Install the requirements: 

`pip install -r requirements.txt`

Go back to the root of the repo: 

`cd ..`

Add `QM9.pkl` to the repo from: https://drive.google.com/file/d/1HpZrkkrR_iqbTLPCOglJh9lsxfc7qYVM/view?usp=share_link

Test the MPNN (if you want to use GPUs run the following command in the terminal `export CUDA_VISIBLE_DEVICES=1` to use the A30 on Boltzmann): 

`python train_MPNN.py`
