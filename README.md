# gcn-qm9-cmu10617
Final team project repo for CMU 10-617 (Fall 2022)
# Builing the environent
First create and activate the environment: `conda create -n torchdrug python=3.9; conda activate torchdrug`

Next install pytorch `pip install torch`

Now install torch-geometric and torch-scatter `pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html`

Clone this repo: `git clone https://github.com/SijieFu/gcn-qm9-cmu10617.git; cd gcn-qm9-cmu10617.git/torchdrug`

Install the requirements: `pip install -r requirements.txt`

Add `QM9.pkl` to the repo from: https://drive.google.com/file/d/1HpZrkkrR_iqbTLPCOglJh9lsxfc7qYVM/view?usp=share_link
