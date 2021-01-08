# Knowledge-Embedding-with-Double-Loss
## Introduction
- OpenKE-DL
  + TransE, TransE-RS, TransE-SS
  + TransH, TransH-RS, TransH-SS
  + TransD, TransD-RS, TransD-SS
  + Complex, Complex-SS
- ProjE-DL
  + ProjE, ProjE-SS
## Training
- Training in OpenKE-DL

```
cd OpenKE-DL
```

Running the following to train:

TransE-SS:
```
python train_transe.py --mode SS --in_path WN18/ --exp {Your experiment name} --u1 6.0 --u2 8.0 --lam 3.0 --lr 0.01 --times 3000 --batch_szie 1200 --dim 100 
```
```
python train_transe.py --mode SS --in_path FB15K/ --exp {Your experiment name} --u1 6.0 --u2 8.0 --lam 2.0 --lr 0.001 --times 3000 --batch_szie 960 --dim 100 
```
Note: u1 is threshold of positive examples, u2 is the threshold of negative examples, --in_path is the path of dataset, mode can be {'MR', 'RS', 'SS'}


TransD-SS:
```
python train_transd.py --mode SS --in_path WN18/ --exp {Your experiment name} --u1 4.0 --u2 8.0 --lam 3.0 --lr 0.01 --times 3000 --batch_szie 1200 --dim 100 
```
```
python train_transd.py --mode SS --in_path FB15K/ --exp {Your experiment name} --u1 8.0 --u2 9.0 --lam 2.0 --lr 0.001 --times 3000 --batch_szie 960 --dim 100 
```


TransH-SS:
```
python train_transh.py --mode SS --in_path WN18/ --exp {Your experiment name} --u1 3.0 --u2 7.0 --lam 3.0 --lr 0.01 --times 3000 --batch_szie 1200 --dim 100 --gpu 0
```
```
python train_transh.py --mode SS --in_path FB15K/ --exp {Your experiment name} --u1 7.0 --u2 8.0 --lam 2.0 --lr 0.001 --times 3000 --batch_szie 960 --dim 100 --gpu 0
```

Complex-SS with sigmoid:
```
python train_complex.py --mode SS --in_path WN18/ --exp {Your experiment name} --u1 0.8 --u2 0.4 --lam 1.0 --lr 0.0001 --times 600 --batch_szie 1200 --dim 100 --gpu 0
```
```
python train_complex.py --mode SS --in_path FB15K/ --exp {Your experiment name} --u1 0.9 --u2 0.7 --lam 1.0 --lr 0.0001 --times 1000 --batch_szie 960 --dim 100 --gpu 0
```

Complex-SS with softmax:
```
python train_complex_softmax.py --mode SS --in_path WN18/ --exp {Your experiment name} --u1 0.8 --u2 0.4 --lam 1.0 --lr 0.0001 --times 600 --batch_szie 1200 --dim 100 --gpu 0
```
```
python train_complex_softmax.py --mode SS --in_path FB15K/ --exp {Your experiment name} --u1 0.9 --u2 0.7 --lam 1.0 --lr 0.0001 --times 1000 --batch_szie 960 --dim 100 --gpu 0
```

- Training in OpenKE-DL

Running the following to train:

```
cd ProjE-DL
```

ProjE-SS:
```
python ProjE.py --lr 0.001 --data ./FB15K/ --neg_weight 0.25 --thr 0.99 -- max_iter 1000 --model FB15K --keep 0.8
```
```
python ProjE.py --lr 0.001 --data ./WN18/ --neg_weight 0.1 --thr 0.99 -- max_iter 1000 --model WN18 --keep 0.8
```
