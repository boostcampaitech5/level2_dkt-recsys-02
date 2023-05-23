import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn.functional import sigmoid
import wandb
from dkt import trainer
from .utils import get_logger, logging_conf
import pdb
import yaml
import gc
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow_addons as tfa
import tqdm

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.augmentations import ClassificationSMOTE
from dataclasses import dataclass
from catboost import CatBoostClassifier, Pool
from wandb.catboost import WandbCallback

logger = get_logger(logger_conf=logging_conf)

@dataclass
class tabnet_custom_callback(Callback):
    def __post_init__(self):
        super().__init__()
    
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        print('---------Training---------')

    def on_train_end(self, logs=None):
        #print(f'-----{self.trainer.feature_importances_}-----')
        wandb.log(dict(epoch=self.trainer.best_epoch,
                       val_auc=self.trainer.best_cost,
                       ))

def config2args(args):
    temp = vars(args)
    for key, value in dict(wandb.config).items():
        temp[key] = value
    return args

def tabnet(args,
        train_data,
        valid_data,
        test_data, 
        categorical_dims 
    ):
    
    wandb.init(project="TabNet", config=vars(args))
    
    if args.sweep_run:
        config2args(args)
    
    #if args.data_augmentation:
    
    #assessmentItemID_list, testId_list, KnowledgeTag_list, answerCode_list = _data_augmentation(args, train_data)
    
    #train_data['assessmentItemID'] = assessmentItemID_list
    #train_data['testId'] = testId_list
    #train_data['KnowledgeTag'] = KnowledgeTag_list
    #train_data['answerCode'] = answerCode_list
    
    X_train = train_data.drop(['answerCode'], axis = 1)
    y_train = train_data[['answerCode']]
    X_valid = valid_data.drop(['answerCode'], axis = 1)
    y_valid = valid_data[['answerCode']]
    
    X_test = test_data.drop(['answerCode'], axis = 1)
    y_test = test_data[['answerCode']].values
    
    test = X_test[X_test['userID'] != X_test['userID'].shift(-1)].to_numpy()
    
    categorical_columns = ['assessmentItemID', 'testId', 'KnowledgeTag']
    
    features = [ col for col in X_train.columns] 

    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
    
    clf = TabNetClassifier(
        n_d = args.hidden_dim,
        n_a = args.hidden_dim,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=args.cat_emb_dim,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={
            "lr":args.lr,
        },
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params={
            'factor': 0.5,
            'mode': "max",
            'verbose': True,
        },
        mask_type='sparsemax', # "sparsemax", entmax
        verbose=1, # 0으로 두면 progress bar 안 나옴
        device_name='cuda',
        seed = args.seed,
        clip_value = args.clip_grad
    )

    #tqdm_callback = tfa.callbacks.TQDMProgressBar() # For progress bar
    
    clf.fit(
        X_train=X_train[features].values, y_train=y_train.values.flatten(),
        eval_set=[(X_train[features].values, y_train.values.flatten()), (X_valid[features].values, y_valid.values.flatten())],
        eval_name=['train', 'valid'],
        eval_metric=['accuracy', 'auc'],
        weights = args.weights,
        max_epochs=20,
        patience=args.patience,
        batch_size=args.batch_size,
        virtual_batch_size=(int)(args.batch_size / 8),
        drop_last=False,
        num_workers=args.num_workers,
        callbacks = [tabnet_custom_callback()],
        augmentations = ClassificationSMOTE(),
        
    )
    
    '''
    print('---------Predict---------')
    preds = clf.predict_proba(test)
    total_preds = preds[:, 1]
    
    print('---------Submission---------')
    write_path = os.path.join(args.output_dir, "submission.csv")
    os.makedirs(name=args.output_dir, exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)
    '''
    
def catboost(args,
        train_data,
        valid_data,
        test_data
    ):
    
    wandb.init(project="catboost", config=vars(args))
    
    if args.sweep_run:
        config2args(args)
    
    X_train = train_data.drop(['answerCode'], axis = 1)
    y_train = train_data[['answerCode']]
    X_valid = valid_data.drop(['answerCode'], axis = 1)
    y_valid = valid_data[['answerCode']]
    
    X_test = test_data.drop(['answerCode'], axis = 1)
    y_test = test_data[['answerCode']].values
    
    test = X_test[X_test['userID'] != X_test['userID'].shift(-1)].to_numpy()
    
    categorical_columns = ['assessmentItemID', 'testId', 'KnowledgeTag']
    
    features = [ col for col in X_train.columns] 

    train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_columns)
    valid_pool = Pool(data=X_valid, label=y_valid, cat_features=categorical_columns)
    
    model = CatBoostClassifier(
        iterations = 30,
        learning_rate = args.lr,
        random_seed = args.seed,
        eval_metric ='AUC',
        early_stopping_rounds = args.patience,
        use_best_model = True,
        cat_features = categorical_columns,
        verbose = 1
    )

    tqdm_callback = tfa.callbacks.TQDMProgressBar() # For progress bar
    
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=20,
        callbacks = [
            WandbCallback()
        ]
    )
    
    WandbCallback.log_summary(model)
    WandbCallback._log_feature_importance(model)
    
    #feature_importance = model.feature_importances_
    #print(feature_importance)
    
def _data_augmentation(args, data):
    grouped_df = data.groupby('userID')
    
    ########마지막 seq를 제외하고는 섞기
    def shuffle(data_list, data):
        last = data[-1]
        before = data[:-1]
        data_list.append(data)
        for i in range(args.shuffle_n):
            # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
            #random_index = np.random.permutation(len(before))
            #shuffled = before[random_index]
            np.random.seed(i)
            shuffled = np.random.permutation(before)
            shuffled = np.append(shuffled, last)
            data_list.append(shuffled)
        np.random.seed(42)
        return data_list
    
    ######FE시에 추가해야함
    assessmentItemID_list = []
    testId_list = []
    KnowledgeTag_list = []
    answerCode_list = []
    #New Feature_list = []
    #userID_list = []
    print('---------Applying Sliding Window---------')
    for userID, user_seq in grouped_df:
        assessmentItemID = user_seq['assessmentItemID'].values[::-1]
        testId = user_seq['testId'].values[::-1]
        KnowledgeTag = user_seq['KnowledgeTag'].values[::-1]
        answerCode = user_seq['answerCode'].values[::-1]
        #New Feature = user_seq['New Feature'].values[::-1]

        start_idx = 0
        if len(user_seq) <= args.max_seq_len:
            ######FE시에 추가해야함
            if args.shuffle_data:
                assessmentItemID_list = shuffle(assessmentItemID_list,  assessmentItemID[::-1])
                testId_list = shuffle(testId_list,  testId[::-1])
                KnowledgeTag_list = shuffle(KnowledgeTag_list,  KnowledgeTag[::-1])
                answerCode_list = shuffle(answerCode_list,  answerCode[::-1])
                #New Feature_list = self.shuffle(New Feature_list,  New Feature[::-1])
            else:
                assessmentItemID_list.append(assessmentItemID[::-1])
                testId_list.append(testId[::-1])
                KnowledgeTag_list.append(KnowledgeTag[::-1])
                answerCode_list.append(answerCode[::-1])
                #New Feature_list.append(New Feature[::-1])
            #userID_list.append([userID]* len(answerCode[::-1]))
        else:
            stop = False
            while stop == False:
                ######FE시에 추가해야함
                if len(answerCode[start_idx: start_idx + args.max_seq_len]) < args.max_seq_len: stop = True
                ######FE시에 추가해야함
                if args.shuffle_data:
                    assessmentItemID_list = shuffle(assessmentItemID_list,  assessmentItemID[start_idx: start_idx + args.max_seq_len][::-1])
                    testId_list = shuffle(testId_list,  testId[start_idx: start_idx + args.max_seq_len][::-1])
                    KnowledgeTag_list = shuffle(KnowledgeTag_list,  KnowledgeTag[start_idx: start_idx + args.max_seq_len][::-1])
                    answerCode_list = shuffle(answerCode_list,  answerCode[start_idx: start_idx + args.max_seq_len][::-1])
                    #New Feature_list = self.shuffle(New Feature_list,  New Feature[start_idx: start_idx + self.max_seq_len][::-1])
                else:
                    assessmentItemID_list.append(assessmentItemID[start_idx: start_idx + args.max_seq_len][::-1])
                    testId_list.append(testId[start_idx: start_idx + args.max_seq_len][::-1])
                    KnowledgeTag_list.append(KnowledgeTag[start_idx: start_idx + args.max_seq_len][::-1])
                    answerCode_list.append(answerCode[start_idx: start_idx + args.max_seq_len][::-1])
                    #New Feature_list.append(New Feature[start_idx: start_idx + self.max_seq_len][::-1])
                #userID_list.append([userID]* len(answerCode[::-1]))
                start_idx += args.window

    ######FE시에 추가해야함
    return assessmentItemID_list, testId_list, KnowledgeTag_list, answerCode_list #New Feature_list
