import os
import argparse

import torch
import wandb

from lightgcn.args import parse_args
from lightgcn.datasets import prepare_dataset
from lightgcn import trainer
from lightgcn.utils import get_logger, set_seeds, logging_conf
import pdb
import pickle

logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    wandb.login()
    wandb.init(project="dkt", config=vars(args))
    set_seeds(args.seed)
    
    use_cuda: bool = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info("Preparing data ...")
    train_data, test_data, id2index  = prepare_dataset(device=device, data_dir=args.data_dir)
    n_node = len(id2index)

    logger.info("Building Model ...")
    model = trainer.build(
        n_node=n_node,
        embedding_dim=args.hidden_dim,
        num_layers=args.n_layers,
        alpha=args.alpha,
    )
    model = model.to(device)
    
    logger.info("Start Training ...")
    graph_emb = trainer.run(
        model=model,
        train_data=train_data,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        model_dir=args.model_dir,
        )
    
    try:
        with open('/opt/ml/input/code/dkt/models_param/feature_mapper.pkl', 'rb') as f: feature_maping_info = pickle.load(f)
    except:
        print('Run dkt train.py first to get feature mapping info')
        raise Exception
    
    user_emb = {}
    item_emb = {}
    for id, index in id2index.items():
        
        if type(id) == int: 
            #user_emb[feature_maping_info['userID'][id]] = graph_emb[index]
            1
        else:
            item_emb[feature_maping_info['assessmentItemID'][id]] = graph_emb[index]


    with open(f'/opt/ml/input/code/lightgcn/models_param/lgcn_item_emb_{args.hidden_dim}.pkl', 'wb') as f: pickle.dump(item_emb, f)
    #with open('/opt/ml/input/code/lightgcn/models_param/lgcn_user_emb.pkl', 'wb') as f: pickle.dump(user_emb, f)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
