# gcn-qm9-cmu10617
Final team project repo for CMU 10-617 (Fall 2022)
# Clone the repo
`git clone https://github.com/SijieFu/gcn-qm9-cmu10617.git`
# Building the environent
First create and activate the environment: 

`conda create -n torchdrug python=3.9; conda activate torchdrug`

Next install pytorch (Linux): 

`conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge`

Now install pytorch-scatter and pytorch-cluster (Linux): 

`conda install pytorch-scatter -c pyg`

`conda install pytorch-cluster -c pyg`

Install the requirements: 

`pip install -r requirements.txt`
# Training a model with `main.py`
