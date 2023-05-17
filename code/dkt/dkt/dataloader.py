import os
import random
import time
from datetime import datetime
from typing import Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import pdb


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.kf = args.kfold

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self,
                   data: np.ndarray,
                   ratio: float = 0.7,
                   shuffle: bool = True,
                   seed: int = 0) -> Tuple[np.ndarray]:
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            #random.shuffle(data)
            data = data.sample(frac=1).reset_index(drop=True)
            
        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]
        return data_1, data_2
    
    def split_data_df(self,
                   data: np.ndarray,
                   ratio: float = 0.7,
                   shuffle: bool = True,
                   seed: int = 0) -> Tuple[np.ndarray]:
        """
        split data into two parts with a given ratio.
        """
        if self.kf:
            kfolds = []
            kf = KFold(n_splits = self.args.n_folds, random_state = self.args.seed, shuffle = True)
            for idx, (train_user, valid_user) in enumerate(kf.split(self.user_list)):
                tmp = {}
                train = data[data['userID'].isin(train_user)]
                train = train.reset_index()
                valid = data[data['userID'].isin(valid_user)]
                valid = valid.reset_index()
                tmp['train'] = train
                tmp['val'] = valid
                kfolds.append(tmp)
            return kfolds
        else:
            train = int(len(self.user_list) * ratio)
            random.shuffle(self.user_list)
            train_user = self.user_list[:train]
            valid_user = self.user_list[train:]
            train = data[data['userID'].isin(train_user)]
            train = train.reset_index()
            valid = data[data['userID'].isin(valid_user)]
            valid = valid.reset_index()
        return train, valid
    
    def __save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        print('---------Preprocessing Data---------')
        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in tqdm(cate_cols):
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s: str):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)
        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Fill in if needed
        # print('---------Feature Engineering---------')
        # df['month'] = df["Timestamp"].str.replace('[^0-9]','', regex=True).map(lambda x: int(x[4:6]))
        # df['day'] = df["Timestamp"].str.replace('[^0-9]','', regex=True).map(lambda x: int(x[6:8]))
        # df['hour'] = df["Timestamp"].str.replace('[^0-9]','', regex=True).map(lambda x: int(x[8:10]))
        # df['minute'] = df["Timestamp"].str.replace('[^0-9]','', regex=True).map(lambda x: int(x[10:12]))
        # df['second'] = df["Timestamp"].str.replace('[^0-9]','', regex=True).map(lambda x: int(x[12:14]))

        return df

    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_tests = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tags = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )
#################
        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                )
            )
        )

        return df

    def load_train_data(self, file_name: str) -> None:
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name: str) -> None:
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, args):
        self.args = args
        self.data = data
        self.max_seq_len = args.max_seq_len
        self.use_past_present = args.past_present
        #######FE시에 추가해야함
        self.grouped_df = self.data.groupby('userID')

        self.grouped_value = (self.grouped_df.apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                    #r[New Feature].values,
                )
            )
        ).values
        
        self.user_list = self.data['userID'].unique().tolist()
        self.window = self.args.window
        self.data_augmentation = self.args.data_augmentation
        #######Sliding Window 적용해 데이터 증가, FE 시에 feature 추가해야함
        if self.data_augmentation:
            self.assessmentItemID_list, self.testId_list, self.KnowledgeTag_list, self.answerCode_list = self._data_augmentation()

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]
        
        # Load from data
        test, question, tag, correct = row[0], row[1], row[2], row[3]
        data = {
            "test": torch.tensor(test + 1, dtype=torch.int),
            "question": torch.tensor(question + 1, dtype=torch.int),
            "tag": torch.tensor(tag + 1, dtype=torch.int),
            "correct": torch.tensor(correct, dtype=torch.int),
        }

        # Generate mask: max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len:]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len-seq_len:] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        data["mask"] = mask
        
####################Generate interaction
        interaction = data["answerCode"] + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1)
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        data["interaction"] = interaction
        data = {feature: feature_seq.int() for feature, feature_seq in data.items()}
        return data

    def __len__(self) -> int:
        if self.data_augmentation: return len(self.answerCode_list)
        else: return len(self.grouped_value)
    
    def _data_augmentation(self):
        ######FE시에 추가해야함
        assessmentItemID_list = []
        testId_list = []
        KnowledgeTag_list = []
        answerCode_list = []

        # New Feature_list = []
        print('---------Applying Sliding Window---------')
        for userID, user_seq in tqdm(self.grouped_df):
            assessmentItemID = user_seq['assessmentItemID'].values[::-1]
            testId = user_seq['testId'].values[::-1]
            KnowledgeTag = user_seq['KnowledgeTag'].values[::-1]
            answerCode = user_seq['answerCode'].values[::-1]
            #New Feature = user_seq['New Feature'].values[::-1]

            start_idx = 0
            if len(user_seq) <= self.max_seq_len:
                ######FE시에 추가해야함
                assessmentItemID_list.append(assessmentItemID[::-1])
                testId_list.append(testId[::-1])
                KnowledgeTag_list.append(KnowledgeTag[::-1])
                answerCode_list.append(answerCode[::-1])
                #New Feature_list.append(New Feature[::-1])

            else:
                while True:

                    ######FE시에 추가해야함
                    if len(answerCode[start_idx: start_idx + self.max_seq_len]) < self.max_seq_len:
                        assessmentItemID_list.append(assessmentItemID[start_idx: start_idx + self.max_seq_len:][::-1])
                        testId_list.append(testId[start_idx: start_idx + self.max_seq_len][::-1])
                        KnowledgeTag_list.append(KnowledgeTag[start_idx: start_idx + self.max_seq_len][::-1])
                        answerCode_list.append(answerCode[start_idx: start_idx + self.max_seq_len][::-1])
                        #New Feature_list.append(New Feature[start_idx: start_idx + self.max_seq_len][::-1])

                        break

                    ######FE시에 추가해야함
                    assessmentItemID_list.append(assessmentItemID[start_idx: start_idx + self.max_seq_len][::-1])
                    testId_list.append(testId[start_idx: start_idx + self.max_seq_len][::-1])
                    KnowledgeTag_list.append(KnowledgeTag[start_idx: start_idx + self.max_seq_len][::-1])
                    answerCode_list.append(answerCode[start_idx: start_idx + self.max_seq_len][::-1])
                    #New Feature_list.append(New Feature[start_idx: start_idx + self.max_seq_len][::-1])
                    start_idx += self.window

        ######FE시에 추가해야함
        return assessmentItemID_list, testId_list, KnowledgeTag_list, answerCode_list #New Feature_list

def get_loaders(args, train: np.ndarray, valid: np.ndarray) -> Tuple[torch.utils.data.DataLoader]:
    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader
