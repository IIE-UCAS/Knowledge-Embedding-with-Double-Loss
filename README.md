# Knowledge-Embedding-with-Double-Loss
## Introduction
- OpenKE-DL
  + TransE
  + TransE-RS
  + TransE-SS
  + TransH
  + TransH-RS
  + TransH-SS
  + TransD
  + TransD-RS
  + TransD-SS
  + Complex
  + Complex-SS
- ProjE-DL
  + ProjE-SS
## Training
- Training in OpenKE-DL

cd OpenKE-DL

Running the following to train:

TransE-SS:
```
python train_transe.py --mode SS --in_path WN18/ --exp {Your experiment name} --u1 4.0 --u2 8.0 --lam 3.0 --lr 0.01 --times 3000 --batch_szie 1200 --dim 100 --gpu 0
```
```
python train_transe.py --mode SS --in_path FB15K/ --exp {Your experiment name} --u1 8.0 --u2 9.0 --lam 2.0 --lr 0.001 --times 3000 --batch_szie 960 --dim 100 --gpu 0
```
Note: u1 is threshold of positive examples, u2 is the threshold of negative examples, --in_path is the path of dataset


TransD-SS:
```
python train_transd.py --mode SS --in_path WN18/ --exp {Your experiment name} --u1 4.0 --u2 8.0 --lam 3.0 --lr 0.01 --times 3000 --batch_szie 1200 --dim 100 --gpu 0
```
```
python train_transd.py --mode SS --in_path FB15K/ --exp {Your experiment name} --u1 8.0 --u2 9.0 --lam 2.0 --lr 0.001 --times 3000 --batch_szie 960 --dim 100 --gpu 0


 
