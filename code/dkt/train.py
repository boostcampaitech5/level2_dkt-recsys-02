import os
import pandas as pd
import numpy as np
import torch
import wandb
import pdb
from dkt import trainer
from dkt.args import parse_args
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, set_seeds, logging_conf
from functools import partial
import yaml

logger = get_logger(logging_conf)


def main(args):
    wandb.login()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)

    train_data = preprocess.get_train_data()
    
    if args.kfold:
        kfolds = preprocess.split_data_df(data=train_data)
    else:
        train_data, valid_data = preprocess.split_data_df(data=train_data)
    
    wandb.init(project="dkt2", config=vars(args))
    
    logger.info("Building Model ...")
    
    
    logger.info("Start Training ...")
    if args.kfold:
        trainer.run_kfold(args=args, kfolds = kfolds)
    else:
        model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
        trainer.run(args=args, train_data=train_data, valid_data=valid_data, model=model)


def config2args(args):
    temp = vars(args)
    for key, value in dict(wandb.config).items():
        temp[key] = value
    return args

def graph_name_parser(args):
    graph_name = ""
    graph_name += f"{args.model}"
    for key, value in dict(wandb.config).items():
        graph_name += f"||{key}:{str(value)}"

    return graph_name

def sweep_main(args):
    
    wandb.init(entity='recommy')
    config2args(args)
    wandb.run.name = graph_name_parser(args)

    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: pd.Dataframe = preprocess.get_train_data()
    
    train_data, valid_data = preprocess.split_data_df(data=train_data)    
    
    logger.info("Building Model ...")
    model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
    
    logger.info("Start Training ...")
    trainer.run(args=args, train_data=train_data, valid_data=valid_data, model=model)
    

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    if args.sweep_run:
        curr_dir = __file__[:__file__.rfind('/')+1]
        with open(curr_dir + 'sweep_config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        file.close()

        sweep_id = wandb.sweep(
            sweep=config[args.model.lower()],
            project='sequence model'
        )
        wandb_train_func = partial(sweep_main, args)
        wandb.agent(sweep_id, function=wandb_train_func, count=args.tuning_count)
    else:
        main(args)
