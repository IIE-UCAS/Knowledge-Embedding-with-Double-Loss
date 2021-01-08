import openke
from openke.config import Trainer, Tester
from openke.module.model import ComplEx
from openke.module.loss import SoftplusLoss
from openke.module.loss import SoftplusSSLossoft
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse
import os
import random
import json
import torch
import numpy as np

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train Complex')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--in_path', type=str, default='FB15K/')
    parser.add_argument('--times', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=960)
    parser.add_argument('--bern_flag', type=int, default=1)
    parser.add_argument('--filter_flag', type=int, default=1)
    parser.add_argument('--rate', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--model', type=str, default='Complex')
    parser.add_argument('--mode', type=str, default='MR')
    parser.add_argument('--exp', type=str, default='0')
    parser.add_argument('--margin', type=float, default=5.0)
    parser.add_argument('--u1', type=float, default=8.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--u2', type=float, default=9.0)
    parser.add_argument('--lam', type=float, default=2.0)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    filename = args.model + '-' + args.mode + '-' + args.exp
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/"+args.in_path,
        batch_size=args.batch_size,
        threads=8,
        sampling_mode="normal",
        bern_flag=args.bern_flag,
        filter_flag=args.filter_flag,
        neg_ent=args.rate,
        neg_rel=0)
    # dataloader for test
    test_dataloader = TestDataLoader("./benchmarks/"+args.in_path, "link")

    # define the model
    complex = ComplEx(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim
    )

    # define the loss function
    if args.mode == 'MR':
        model = NegativeSampling(
            model=complex,
            loss=SoftplusLoss(),
            batch_size=train_dataloader.get_batch_size(),
            regul_rate=args.lam
        )
    elif args.mode == 'SS':
        model = NegativeSampling(
            model=complex,
            loss=SoftplusSSLossoft(args.u1, args.u2),
            batch_size=train_dataloader.get_batch_size(),
            regul_rate=args.lam
        )

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=args.times, alpha=args.lr, use_gpu=True, opt_method="adam")
    trainer.run()
    complex.save_checkpoint('./checkpoint/complex'+args.exp+'.ckpt')

    # test the model
    complex.load_checkpoint('./checkpoint/complex'+args.exp+'.ckpt')
    tester = Tester(model=complex, data_loader=test_dataloader, use_gpu=True)
    mrr, mr, hit10, hit3, hit1, mrr_raw, mr_raw, hit10_raw, hit3_raw, hit1_raw = tester.run_link_prediction(type_constrain=False)
    d = {}
    d['mrr'] = mrr
    d['mr'] = mr
    d['hit10'] = hit10
    d['hit3'] = hit3
    d['hit1'] = hit1
    d['args'] = vars(args)
    d['mrr_raw'] = mrr_raw
    d['mr_raw'] = mr_raw
    d['hit10_raw'] = hit10_raw
    d['hit3_raw'] = hit3_raw
    d['hit1_raw'] = hit1_raw
    with open('./log/' + filename+'.json', 'w') as f:
        json.dump(d, f)
