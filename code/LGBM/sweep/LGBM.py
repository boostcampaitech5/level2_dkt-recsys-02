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
import seaborn as sns

#modul
from feature import feature_engineering,elo



args = parse_args()
    
# wandb setting
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
fold_len = 5
gkf = GroupKFold(n_splits = 5)

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
wandb.finish()
print(f"k-fold valid auc: {result_auc/fold_len} , k-fold valid acc: {result_acc/fold_len}")
    
