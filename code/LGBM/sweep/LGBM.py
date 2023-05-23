from args import parse_args
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#processing
from sklearn.model_selection import KFold, GroupKFold

#model
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

#else
import pdb
import wandb
import yaml
from functools import partial
#modul
from feature import feature_engineering,elo

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

def main(args):
    # wandb setting
    if args.sweep_run:
        wandb.init()
        config2args(args)
        wandb.run.name = graph_name_parser(args)

    else:
        wandb.login()
        wandb.init(project="LGBM",entity = 'recommy', config=vars(args))
        wandb.run.name = f"LGBM_{args.learning_rate}_num_leaves_{args.num_leaves}_feature_fraction_{args.feature_fraction}_bagging_fraction_{args.bagging_fraction}_bagging_freq_{args.bagging_freq}"
        wandb.run.save()


    # 1. 데이터로딩
    dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
    }  
    DATA_PATH = '/opt/ml/input/data/'
    df = pd.read_csv(DATA_PATH+'train_data.csv' , dtype=dtype, parse_dates=['Timestamp'])

    #2. FE
    df = feature_engineering(df)
    # elo관련 피쳐 3개 생성
    for col in ['assessmentItemID','testId','KnowledgeTag']:
        df = elo(df,col)

    # 사용할 Feature 설정
    FEATS = [
            'KnowledgeTag', 
            'user_correct_answer', 
            'user_total_answer', 
            'user_acc',
            'test_mean', 
            'test_sum', 
            'tag_mean',
            'tag_sum',
            'elapsed',
            'elapsed_cumsum',
            'month',
            'day',
            'hour',
            'dayname',
            'elapsed_med',
            'bigclass',
            'bigclasstime',
            'bigclass_acc',
            'bigclass_sum',
            'bigclass_count',
            'elo_assessmentItemID',
            'elo_testId',
            'elo_KnowledgeTag'
                ]


    params = {
    # "max_depth": args.max_depth, # default=-1 (no limit)
    "learning_rate": args.learning_rate,  # default = 0.1, [0.0005 ~ 0.5]
    "boosting": "gbdt",
    "objective": args.objective,
    "metric": args.metric,
    "num_leaves": args.num_leaves,  # default = 31, [10, 20, 31, 40, 50]
    "feature_fraction": args.feature_fraction,  # default = 1.0, [0.4, 0.6, 0.8, 1.0]
    "bagging_fraction": args.bagging_fraction,  # default = 1.0, [0.4, 0.6, 0.8, 1.0]
    "bagging_freq": args.bagging_freq,  # default = 0, [0, 1, 2, 3, 4]
    "seed": 42,
    "verbose": -1,
    }

    # 3. GroupKfold & train

    y_train = df['answerCode']
    train = df.drop(['answerCode'], axis=1)

    groups = train['userID']
    fold_len = 2
    gkf = GroupKFold(n_splits = 2)

    k_auc_list = []
    result_auc = 0
    result_acc = 0
    for i,(train_index, test_index) in enumerate(gkf.split(train,y_train,groups= groups)):
        lgb_train = lgb.Dataset(train.iloc[train_index][FEATS],y_train.iloc[train_index])
        lgb_test = lgb.Dataset(train.iloc[test_index][FEATS],y_train.iloc[test_index])


        model = lgb.train(
        params, 
        lgb_train,
        valid_sets=[lgb_train,lgb_test],
        verbose_eval=100,
        early_stopping_rounds=100,
        num_boost_round=500,
        callbacks=[wandb.lightgbm.wandb_callback()]
    )
        wandb.lightgbm.log_summary(model, save_model_checkpoint=True)

        preds = model.predict(train.iloc[test_index][FEATS])
        acc = accuracy_score(y_train.iloc[test_index], np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(y_train.iloc[test_index], preds)
        result_auc+=auc
        result_acc+=acc
        metric = {"valid_accuracy" : acc,
                "valid_roc_auc" : auc
                }
        wandb.log(metric)

        print(f"--------------------K-fold {i}--------------------")
        print(f'VALID AUC : {auc} ACC : {acc}\n')
    kfold_auc = result_auc/fold_len
    
    wandb.log({"kfold_auc" : kfold_auc})
    
    if args.sweep_run:
        curr_dir = __file__[:__file__.rfind('/')+1]
        with open(curr_dir + 'LGBM_best_auc.yaml') as file:
            output = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
            
        if output[args.model.lower()]['best_auc'] < kfold_auc:
            output[args.model.lower()]['best_auc'] = float(kfold_auc)
            output[args.model.lower()]['parameter'] = dict(zip(dict(wandb.config).keys(),map(lambda x: x if type(x) == str else float(x) , dict(wandb.config).values())))
            
        with open(curr_dir + 'LGBM_best_auc.yaml', 'w') as file:
            yaml.dump(output, file, default_flow_style=False)
        file.close()

    else:
        wandb.finish()
        
    print(f"k-fold valid auc: {result_auc/fold_len} , k-fold valid acc: {result_acc/fold_len}")
        
if __name__ == "__main__":
    args = parse_args()
    
    if args.sweep_run:
        curr_dir = __file__[:__file__.rfind('/')+1]
        with open(curr_dir + 'LGBM_sweep.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        file.close()

        sweep_id = wandb.sweep(
            sweep=config[args.model.lower()],
            project='LGBM'
        )
        wandb_train_func = partial(main, args)
        wandb.agent(sweep_id, function=wandb_train_func, count=args.tuning_count)

    else:
        main(args)
