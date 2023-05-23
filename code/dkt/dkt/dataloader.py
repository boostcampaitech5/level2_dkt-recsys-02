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
from sklearn.model_selection import KFold, GroupKFold
import pdb
from .featureEngineering import feature_engineering,elo
import pickle
import json
from sklearn.preprocessing import StandardScaler

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.kf = args.kfold
        self.cat_dims = {}

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_cat_dims(self):
        return self.cat_dims

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
            #kf =  GroupKFold(n_splits = self.args.n_folds)
            kf = KFold(n_splits = self.args.n_folds, random_state= seed, shuffle= True)
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
######### FE 시에 범주형 변수 추가 시 추가 부탁
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag","question_N","dayname","bigclass"]
        feature_maping_info = {}
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
            label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            feature_maping_info[col] = label_mapping

            self.cat_dims[col] = len(le.classes_)
            
            df[col] = test
        
######### FE 시에  연속형 변수 추가 시 추가 부탁
        numeric_cols = ["user_correct_answer", "user_total_answer", "user_acc",
                    "test_mean", "test_sum", "tag_mean", "tag_sum",
                    "elapsed", "elapsed_cumsum", "month", "day", "hour",
                    "elapsed_med", "bigclasstime", "bigclass_acc",
                    "bigclass_sum", "bigclass_count", "elo"]
        
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        def convert_time(s: str):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)
        
        curr_dir = __file__[:__file__.rfind('/')+1]
        with open(curr_dir + '../models_paramfeature_mapper.pkl', 'wb') as f: 
            pickle.dump(feature_maping_info, f)
        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Fill in if needed
        print('---------Feature Engineering---------')
        ######## Feature별 unique한 값의 개수를 저장
        num_feature = {}

        ########Category 
        df['question_N'] = df['assessmentItemID'].apply(lambda x: x[-3:]) ####13개
        num_feature['question_N'] =df['question_N'].nunique()

        #########Continous
        # featureEngineering.py를 import 해서 사용
        df = feature_engineering(df)
        df = elo(df)
        num_feature['dayname'] = df['dayname'].nunique()
        num_feature['bigclass'] = df['bigclass'].nunique()

        # 범주형 변수 : [KnowledgeTag,dayname, bigclass]
        # 추가된 피쳐
        # Feat = 'user_correct_answer', #유저가 문제 푼 횟수
        #  'user_total_answer', #유저가 문제 맞춘 횟수
        #  'user_acc', #유저의 정답률
        #  'test_mean', #문항의 정답률
        #  'test_sum', #문항의 정답횟수
        #  'tag_mean', #태그의 정답률
        #  'tag_sum', #태그의 정답횟수
        #  'elapsed', #유저의 문제풀이시간
        #  'elapsed_cumsum', #유저의 문제풀이시간 누적
        #  'elapsed_med', #유저의 문제풀이시간 중앙값
        #  'month', #월
        #  'day', #일
        #  'hour', #시간
        #  'dayname', #요일
        #  'bigclass', #대분류
        #  'bigclasstime', #대분류별 문제풀이시간
        #  'bigclass_acc', #대분류별 정답률
        #  'bigclass_sum', #대분류별 문제 맞춘 횟수
        #  'bigclass_count', #대분류별 문제 푼 횟수
        #  'elo' #유저의 문제풀이능력

        
        curr_dir = __file__[:__file__.rfind('/')+1]
        with open(curr_dir + '../models_param/num_feature.json', 'w') as f: 
            json.dump(num_feature, f)

        return df

    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)



######### 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
######### 카테고리 변수 FE 시에 추가 부탁
        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_tests = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tags = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )
        self.args.n_daynames = len(
            np.load(os.path.join(self.args.asset_dir, "dayname_classes.npy"))
        )
        self.args.n_bigclass = len(
            np.load(os.path.join(self.args.asset_dir, "bigclass_classes.npy"))
        )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
#########  FE 시에 추가 부탁
        columns = [ "userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag",
                    "user_correct_answer", "user_total_answer", "user_acc",
                    "test_mean", "test_sum", "tag_mean", "tag_sum",
                    "elapsed", "elapsed_cumsum", "month", "day", "hour", "dayname",
                    "elapsed_med", "bigclass", "bigclasstime", "bigclass_acc",
                    "bigclass_sum", "bigclass_count", "elo","question_N"
                ]
        self.user_list = df['userID'].unique().tolist()
        self.userID = df['userID'].unique().tolist()
        self.assessmentItemID = df['assessmentItemID'].unique().tolist()
        self.testId = df['testId'].unique().tolist()
        self.answerCode = df['answerCode'].unique().tolist()
        self.KnowledgeTag = df['KnowledgeTag'].unique().tolist()
        self.question_N = df['question_N'].unique().tolist()
        self.user_correct_answer = df['user_correct_answer'].unique().tolist()
        self.user_total_answer = df['user_total_answer'].unique().tolist()
        self.user_acc = df['user_acc'].unique().tolist()
        self.test_mean = df['test_mean'].unique().tolist()
        self.test_sum = df['test_sum'].unique().tolist()
        self.tag_mean = df['tag_mean'].unique().tolist()
        self.tag_sum = df['tag_sum'].unique().tolist()
        self.elapsed = df['elapsed'].unique().tolist()
        self.elapsed_cumsum = df['elapsed_cumsum'].unique().tolist()
        self.month = df['month'].unique().tolist()
        self.day = df['day'].unique().tolist()
        self.hour = df['hour'].unique().tolist()
        self.dayname = df['dayname'].unique().tolist()
        self.elapsed_med = df['elapsed_med'].unique().tolist()
        self.bigclass = df['bigclass'].unique().tolist()
        self.bigclasstime = df['bigclasstime'].unique().tolist()
        self.bigclass_acc = df['bigclass_acc'].unique().tolist()
        self.bigclass_sum = df['bigclass_sum'].unique().tolist()
        self.bigclass_count = df['bigclass_count'].unique().tolist()
        self.elo = df['elo'].unique().tolist()

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
        self.shuffle_data = args.shuffle_data
        self.shuffle_n = args.shuffle_n
        #######FE시에 추가해야함
        self.grouped_df = self.data.groupby('userID')

        self.grouped_value = (self.grouped_df.apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                    r['question_N'].values,
                    # r[New Feature].values,
                    r['user_correct_answer'].values, 
                    r['user_total_answer'].values, 
                    r['user_acc'].values,
                    r['test_mean'].values, 
                    r['test_sum'].values, 
                    r['tag_mean'].values,
                    r['tag_sum'].values,
                    r['elapsed'].values,
                    r['elapsed_cumsum'].values,
                    r['month'].values,
                    r['hour'].values,
                    r['day'].values,
                    r['dayname'].values,
                    r['elapsed_med'].values,
                    r['bigclass'].values,
                    r['bigclasstime'].values,
                    r['bigclass_acc'].values,
                    r['bigclass_sum'].values,
                    r['bigclass_count'].values,
                    r['elo'].values
                 )
            )
        ).values
        
        self.user_list = self.data['userID'].unique().tolist()
        self.window = self.args.window
        self.data_augmentation = self.args.data_augmentation
        #######Sliding Window 적용해 데이터 증가, FE 시에 feature 추가해야함
        if self.data_augmentation:
            self.assessmentItemID_list, self.testId_list, self.KnowledgeTag_list, self.answerCode_list, self.question_N_list, self.user_correct_answer_list, self.user_total_answer_list, self.user_acc_list, self.test_mean_list , self.test_mean_list , self.test_sum_list, self.tag_mean_list, self.tag_sum_list, self.elapsed_list,self.elapsed_cumsum_list, self.month_list, self.hour_list, self.day_list, self.dayname_list, self.elapsed_med_list, self.bigclass_list,self.bigclasstime_list,self.bigclass_acc_list, self.bigclass_sum_list,self.bigclass_count_list, self.elo_list = self._data_augmentation()

    def __getitem__(self, index: int) -> dict:
####################Sliding Window 적용 시


        if self.data_augmentation:
####################FE 추가 시 추가해야함
            assessmentItemID = self.assessmentItemID_list[index]
            testId = self.testId_list[index]
            KnowledgeTag = self.KnowledgeTag_list[index]
            answerCode = self.answerCode_list[index]
            #userID = self.userID_list[index]
            #New Feature = self.New_Feature_list[index]
            question_N = self.question_N_list[index]
            user_correct_answer = self.user_correct_answer_list[index]
            user_total_answer = self.user_total_answer_list[index]
            user_acc = self.user_acc_list[index]
            test_mean = self.test_mean_list[index]
            test_sum = self.test_sum_list[index]
            tag_mean = self.tag_mean_list[index]
            tag_sum = self.tag_sum_list[index]
            elapsed = self.elapsed_list[index]
            elapsed_cumsum = self.elapsed_cumsum_list[index]
            month = self.month_list[index]
            day = self.day_list[index]
            hour = self.hour_list[index]
            dayname = self.dayname_list[index]
            elapsed_med = self.elapsed_med_list[index]
            bigclass = self.bigclass_list[index]
            bigclasstime = self.bigclasstime_list[index]
            bigclass_acc = self.bigclass_acc_list[index]
            bigclass_sum = self.bigclass_sum_list[index]
            bigclass_count = self.bigclass_count_list[index]
            elo = self.elo_list[index]

    
            cat_data = {
            "testId": torch.tensor(testId + 1, dtype=torch.int),
            "assessmentItemID": torch.tensor(assessmentItemID + 1, dtype=torch.int),
            "KnowledgeTag": torch.tensor(KnowledgeTag + 1, dtype=torch.int),
            "answerCode": torch.tensor(answerCode, dtype=torch.int),
            #"userID" : torch.tensor(userID, dtype=torch.int),
            #New Feature = torch.tensor(New Feature + 1, dtype=torch.int)
            "question_N" : torch.tensor(question_N + 1, dtype=torch.int),
            "dayname" : torch.tensor(dayname + 1, dtype=torch.int),
            "bigclass" : torch.tensor(bigclass + 1, dtype = torch.int)
            }

            cont_data = {
                # "New Feature" : torch.tensor(New Feature, dtype=torch.float),
                "user_correct_answer" : torch.tensor(user_correct_answer, dtype=torch.float),
                "user_total_answer" : torch.tensor(user_total_answer, dtype=torch.float),
                "user_acc" : torch.tensor(user_acc, dtype=torch.float),
                "test_mean" : torch.tensor(test_mean, dtype=torch.float),
                "test_sum" : torch.tensor(test_sum, dtype=torch.float),
                "tag_mean" : torch.tensor(tag_mean, dtype=torch.float),
                "tag_sum" : torch.tensor(tag_sum, dtype=torch.float),
                "elapsed" : torch.tensor(elapsed, dtype=torch.float),
                "elapsed_cumsum" : torch.tensor(elapsed_cumsum, dtype=torch.float),
                "month" : torch.tensor(month, dtype=torch.float),
                "day" : torch.tensor(day, dtype=torch.float),
                "hour" : torch.tensor(hour, dtype=torch.float),
                "elapsed_med" : torch.tensor(elapsed_med, dtype=torch.float),
                "bigclasstime" : torch.tensor(bigclasstime, dtype=torch.float),
                "bigclass_acc" : torch.tensor(bigclass_acc, dtype=torch.float),
                "bigclass_sum" : torch.tensor(bigclass_sum, dtype=torch.float),
                "bigclass_count" : torch.tensor(bigclass_count, dtype=torch.float),
                "elo" : torch.tensor(elo, dtype=torch.float),
                
            }
            seq_len = len(answerCode)

####################Mask 만들기
            if seq_len >= self.max_seq_len:
                mask = torch.ones(self.max_seq_len, dtype=torch.int16)
            else:
                for feature in cat_data:
                    # Pre-padding non-valid sequences
                    tmp = torch.zeros(self.max_seq_len)
                    tmp[self.max_seq_len-seq_len:] = cat_data[feature]
                    cat_data[feature] = tmp
                mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
                mask[-seq_len:] = 1

            cat_data["mask"] = mask
            #####cont
            seq_len = len(answerCode)
            
            if seq_len > self.max_seq_len:
                for k, seq in cont_data.items():
                    cont_data[k] = seq[-self.max_seq_len:]
                mask = torch.ones(self.max_seq_len, dtype=torch.int16)
            else:
                for k, seq in cont_data.items():
                    # Pre-padding non-valid sequences
                    tmp = torch.zeros(self.max_seq_len)
                    tmp[self.max_seq_len-seq_len:] = cont_data[k]
                    cont_data[k] = tmp
                mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
                mask[-seq_len:] = 1
            # cont_data["mask"] = mask 
            
####################Sliding Window 미적용 시
        else:
            row = self.grouped_value[index]
####################FE 추가 시 추가해야함
            testId,assessmentItemID, KnowledgeTag, answerCode, question_N, user_correct_answer, user_total_answer,user_acc, test_mean, test_sum, tag_mean, tag_sum, elapsed,elapsed_cumsum, month, hour, day, dayname, elapsed_med, bigclass, bigclasstime, bigclass_acc,bigclass_sum ,bigclass_count, elo = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22], row[23] ,row[24] 

            
            cat_data = {
            "testId": torch.tensor(testId + 1, dtype=torch.int),
            "assessmentItemID": torch.tensor(assessmentItemID + 1, dtype=torch.int),
            "KnowledgeTag": torch.tensor(KnowledgeTag + 1, dtype=torch.int),
            "answerCode": torch.tensor(answerCode, dtype=torch.int),
            #"userID" : torch.tensor(userID, dtype=torch.int),
            #New Feature = torch.tensor(New Feature + 1, dtype=torch.int)
            "question_N" : torch.tensor(question_N + 1, dtype=torch.int),
            "dayname" : torch.tensor(dayname + 1, dtype=torch.int),
            "bigclass" : torch.tensor(bigclass + 1, dtype = torch.int)
            }
            
            cont_data = {
                # "New Feature" : torch.tensor(New Feature, dtype=torch.float),
                "user_correct_answer" : torch.tensor(user_correct_answer, dtype=torch.float),
                "user_total_answer" : torch.tensor(user_total_answer, dtype=torch.float),
                "user_acc" : torch.tensor(user_acc, dtype=torch.float),
                "test_mean" : torch.tensor(test_mean, dtype=torch.float),
                "test_sum" : torch.tensor(test_sum, dtype=torch.float),
                "tag_mean" : torch.tensor(tag_mean, dtype=torch.float),
                "tag_sum" : torch.tensor(tag_sum, dtype=torch.float),
                "elapsed" : torch.tensor(elapsed, dtype=torch.float),
                "elapsed_cumsum" : torch.tensor(elapsed_cumsum, dtype=torch.float),
                "month" : torch.tensor(month, dtype=torch.float),
                "day" : torch.tensor(day, dtype=torch.float),
                "hour" : torch.tensor(hour, dtype=torch.float),
                "elapsed_med" : torch.tensor(elapsed_med, dtype=torch.float),
                "bigclasstime" : torch.tensor(bigclasstime, dtype=torch.float),
                "bigclass_acc" : torch.tensor(bigclass_acc, dtype=torch.float),
                "bigclass_sum" : torch.tensor(bigclass_sum, dtype=torch.float),
                "bigclass_count" : torch.tensor(bigclass_count, dtype=torch.float),
                "elo" : torch.tensor(elo, dtype=torch.float)
                
            }

####################Mask 만들기       
            seq_len = len(answerCode)

            if seq_len > self.max_seq_len:
                for k, seq in cat_data.items():
                    cat_data[k] = seq[-self.max_seq_len:]
                mask = torch.ones(self.max_seq_len, dtype=torch.int16)
            else:
                for k, seq in cat_data.items():
                    # Pre-padding non-valid sequences
                    pdb.set_trace()
                    tmp = torch.zeros(self.max_seq_len)
                    tmp[self.max_seq_len-seq_len:] = cat_data[k]
                    # tmp[self.max_seq_len-seq_len:self.max_seq_len] = cat_data[k]
                    cat_data[k] = tmp
                mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
                mask[-seq_len:] = 1
            cat_data["mask"] = mask 
            #####cont
            seq_len = len(answerCode)
            
            if seq_len > self.max_seq_len:
                for k, seq in cont_data.items():
                    cont_data[k] = seq[-self.max_seq_len:]
                mask = torch.ones(self.max_seq_len, dtype=torch.int16)
            else:
                for k, seq in cont_data.items():
                    # Pre-padding non-valid sequences
                    tmp = torch.zeros(self.max_seq_len)
                    tmp[self.max_seq_len-seq_len:] = cont_data[k]
                    cont_data[k] = tmp
                mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
                mask[-seq_len:] = 1
            # cont_data["mask"] = mask 
        len_cont = {}
        len_cont['n_cont'] = len(cont_data)
        curr_dir = __file__[:__file__.rfind('/')+1]
        with open(curr_dir + '../models_param/len_cont.json', 'w') as f: 
            json.dump(len_cont, f)
        
####################Generate interaction
        interaction = cat_data["answerCode"] + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1)
        interaction_mask = cat_data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        cat_data["interaction"] = interaction
        cat_data = {feature: feature_seq.int() for feature, feature_seq in cat_data.items()}
        
        ##cont
        cont_data = {feature: feature_seq.float() for feature, feature_seq in cont_data.items()}

        data = {}
        data['category'] = cat_data
        data['continous'] = cont_data
        


        return data

    def __len__(self) -> int:
        if self.data_augmentation: return len(self.answerCode_list)
        else: return len(self.grouped_value)

    ########마지막 seq를 제외하고는 섞기
    def shuffle(self, data_list, data):
        last = data[-1]
        before = data[:-1]
        data_list.append(data)
        for i in range(self.shuffle_n):
            # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
            #random_index = np.random.permutation(len(before))
            #shuffled = before[random_index]
            np.random.seed(i)
            shuffled = np.random.permutation(before)
            shuffled = np.append(shuffled, last)
            data_list.append(shuffled)
        np.random.seed(42)
        return data_list
    
    def pad_sequence(self, seq, max_len, padding_value = 0):
        try:
            seq_len, col = seq.shape
            padding = np.zeros((max_len - seq_len, col)) + padding_value
        except:
            seq_len = seq.shape[0]
            padding = np.zeros((max_len - seq_len, )) + padding_value

        padding_seq = np.concatenate([padding, seq])

        return padding_seq

    def _data_augmentation(self):
        ######FE시에 추가해야함
        assessmentItemID_list = []
        testId_list = []
        KnowledgeTag_list = []
        answerCode_list = []
        #New Feature_list = []
        #userID_list = []
        question_N_list = []
        user_correct_answer_list = []
        user_total_answer_list = []
        user_acc_list = []
        test_mean_list = []
        test_sum_list = []
        tag_mean_list = []
        tag_sum_list = []
        elapsed_list = []
        elapsed_cumsum_list = []
        month_list = []
        day_list = []
        hour_list = []
        dayname_list = []
        elapsed_med_list = []
        bigclass_list = []
        bigclasstime_list = []
        bigclass_acc_list = []
        bigclass_sum_list = []
        bigclass_count_list = []
        elo_list = []
        print('---------Applying Sliding Window---------')
        for userID, user_seq in tqdm(self.grouped_df):
            assessmentItemID = user_seq['assessmentItemID'].values[::-1]
            testId = user_seq['testId'].values[::-1]
            KnowledgeTag = user_seq['KnowledgeTag'].values[::-1]
            answerCode = user_seq['answerCode'].values[::-1]
            #New Feature = user_seq['New Feature'].values[::-1]
            question_N = user_seq['question_N'].values[::-1]
            user_correct_answer = user_seq['user_correct_answer'].values[::-1]
            user_total_answer = user_seq['user_total_answer'].values[::-1]
            user_acc = user_seq['user_acc'].values[::-1]
            test_mean = user_seq['test_mean'].values[::-1]
            test_sum = user_seq['test_sum'].values[::-1]
            tag_mean = user_seq['tag_mean'].values[::-1]
            tag_sum = user_seq['tag_sum'].values[::-1]
            elapsed = user_seq['elapsed'].values[::-1]
            elapsed_cumsum = user_seq['elapsed_cumsum'].values[::-1]
            month = user_seq['month'].values[::-1]
            day = user_seq['day'].values[::-1]
            hour = user_seq['hour'].values[::-1]
            dayname = user_seq['dayname'].values[::-1]
            elapsed_med = user_seq['elapsed_med'].values[::-1]
            bigclass = user_seq['bigclass'].values[::-1]
            bigclasstime = user_seq['bigclasstime'].values[::-1]
            bigclass_acc = user_seq['bigclass_acc'].values[::-1]
            bigclass_sum = user_seq['bigclass_sum'].values[::-1]
            bigclass_count = user_seq['bigclass_count'].values[::-1]
            elo = user_seq['elo'].values[::-1]

            start_idx = 0
            if len(user_seq) <= self.max_seq_len:
                ######FE시에 추가해야함
                if self.shuffle_data:
                    assessmentItemID_list = self.shuffle(assessmentItemID_list,  assessmentItemID[::-1])
                    testId_list = self.shuffle(testId_list,  testId[::-1])
                    KnowledgeTag_list = self.shuffle(KnowledgeTag_list,  KnowledgeTag[::-1])
                    answerCode_list = self.shuffle(answerCode_list,  answerCode[::-1])
                    #New Feature_list = self.shuffle(New Feature_list,  New Feature[::-1])      
                    question_N_list = self.shuffle(question_N_list, question_N[::-1])
                    user_correct_answer_list = self.shuffle(user_correct_answer_list, user_correct_answer[::-1])
                    user_total_answer_list = self.shuffle(user_total_answer_list, user_total_answer[::-1])
                    user_acc_list = self.shuffle(user_acc_list, user_acc[::-1])
                    test_mean_list = self.shuffle(test_mean_list, test_mean[::-1])
                    test_sum_list = self.shuffle(test_sum_list, test_sum[::-1])
                    tag_mean_list = self.shuffle(tag_mean_list, tag_mean[::-1])
                    tag_sum_list = self.shuffle(tag_sum_list, tag_sum[::-1])
                    elapsed_list = self.shuffle(elapsed_list, elapsed[::-1])
                    elapsed_cumsum_list = self.shuffle(elapsed_cumsum_list, elapsed_cumsum[::-1])
                    month_list = self.shuffle(month_list, month[::-1])
                    day_list = self.shuffle(day_list, day[::-1])
                    hour_list = self.shuffle(hour_list, hour[::-1])
                    dayname_list = self.shuffle(dayname_list, dayname[::-1])
                    elapsed_med_list = self.shuffle(elapsed_med_list, elapsed_med[::-1])
                    bigclass_list = self.shuffle(bigclass_list, bigclass[::-1])
                    bigclasstime_list = self.shuffle(bigclasstime_list, bigclasstime[::-1])
                    bigclass_acc_list = self.shuffle(bigclass_acc_list, bigclass_acc[::-1])
                    bigclass_sum_list = self.shuffle(bigclass_sum_list, bigclass_sum[::-1])
                    bigclass_count_list = self.shuffle(bigclass_count_list, bigclass_count[::-1])
                    elo_list = self.shuffle(elo_list, elo[::-1])
                else:
                    assessmentItemID_list.append(assessmentItemID[::-1])
                    testId_list.append(testId[::-1])
                    KnowledgeTag_list.append(KnowledgeTag[::-1])
                    answerCode_list.append(answerCode[::-1])
                    #New Feature_list.append(New Feature[::-1])
                    question_N_list.append(question_N[::-1])
                    # userID_list.append([userID]* len(answerCode[::-1])
                    user_correct_answer_list.append(user_correct_answer[::-1])
                    user_total_answer_list.append(user_total_answer[::-1])
                    user_acc_list.append(user_acc[::-1])
                    test_mean_list.append(test_mean[::-1])
                    test_sum_list.append(test_sum[::-1])
                    tag_mean_list.append(tag_mean[::-1])
                    tag_sum_list.append(tag_sum[::-1])
                    elapsed_list.append(elapsed[::-1])
                    elapsed_cumsum_list.append(elapsed_cumsum[::-1])
                    month_list.append(month[::-1])
                    day_list.append(day[::-1])
                    hour_list.append(hour[::-1])
                    dayname_list.append(dayname[::-1])
                    elapsed_med_list.append(elapsed_med[::-1])
                    bigclass_list.append(bigclass[::-1])
                    bigclasstime_list.append(bigclasstime[::-1])
                    bigclass_acc_list.append(bigclass_acc[::-1])
                    bigclass_sum_list.append(bigclass_sum[::-1])
                    bigclass_count_list.append(bigclass_count[::-1])
                    elo_list.append(elo[::-1])
            else:
                stop = False
                while stop == False:
                    ######FE시에 추가해야함
                    if len(answerCode[start_idx: start_idx + self.max_seq_len]) < self.max_seq_len: stop = True
                    ######FE시에 추가해야함
                    if self.shuffle_data:
                        assessmentItemID_list = self.shuffle(assessmentItemID_list,  assessmentItemID[start_idx: start_idx + self.max_seq_len][::-1])
                        testId_list = self.shuffle(testId_list,  testId[start_idx: start_idx + self.max_seq_len][::-1])
                        KnowledgeTag_list = self.shuffle(KnowledgeTag_list,  KnowledgeTag[start_idx: start_idx + self.max_seq_len][::-1])
                        answerCode_list = self.shuffle(answerCode_list,  answerCode[start_idx: start_idx + self.max_seq_len][::-1])
                        #New Feature_list = self.shuffle(New Feature_list,  New Feature[start_idx: start_idx + self.max_seq_len][::-1])
                        question_N_list = self.shuffle(question_N_list, question_N[start_idx: start_idx + self.max_seq_len][::-1])
                        user_correct_answer_list = self.shuffle(user_correct_answer_list, user_correct_answer[start_idx: start_idx + self.max_seq_len][::-1])
                        user_total_answer_list = self.shuffle(user_total_answer_list, user_total_answer[start_idx: start_idx + self.max_seq_len][::-1])
                        user_acc_list = self.shuffle(user_acc_list, user_acc[start_idx: start_idx + self.max_seq_len][::-1])
                        test_mean_list = self.shuffle(test_mean_list, test_mean[start_idx: start_idx + self.max_seq_len][::-1])
                        test_sum_list = self.shuffle(test_sum_list, test_sum[start_idx: start_idx + self.max_seq_len][::-1])
                        tag_mean_list = self.shuffle(tag_mean_list, tag_mean[start_idx: start_idx + self.max_seq_len][::-1])
                        tag_sum_list = self.shuffle(tag_sum_list, tag_sum[start_idx: start_idx + self.max_seq_len][::-1])
                        elapsed_list = self.shuffle(elapsed_list, elapsed[start_idx: start_idx + self.max_seq_len][::-1])
                        elapsed_cumsum_list = self.shuffle(elapsed_cumsum_list, elapsed_cumsum[start_idx: start_idx + self.max_seq_len][::-1])
                        month_list = self.shuffle(month_list, month[start_idx: start_idx + self.max_seq_len][::-1])
                        day_list = self.shuffle(day_list, day[start_idx: start_idx + self.max_seq_len][::-1])
                        hour_list = self.shuffle(hour_list, hour[start_idx: start_idx + self.max_seq_len][::-1])
                        dayname_list = self.shuffle(dayname_list, dayname[start_idx: start_idx + self.max_seq_len][::-1])
                        elapsed_med_list = self.shuffle(elapsed_med_list, elapsed_med[start_idx: start_idx + self.max_seq_len][::-1])
                        bigclass_list = self.shuffle(bigclass_list, bigclass[start_idx: start_idx + self.max_seq_len][::-1])
                        bigclasstime_list = self.shuffle(bigclasstime_list, bigclasstime[start_idx: start_idx + self.max_seq_len][::-1])
                        bigclass_acc_list = self.shuffle(bigclass_acc_list, bigclass_acc[start_idx: start_idx + self.max_seq_len][::-1])
                        bigclass_sum_list = self.shuffle(bigclass_sum_list, bigclass_sum[start_idx: start_idx + self.max_seq_len][::-1])
                        bigclass_count_list = self.shuffle(bigclass_count_list, bigclass_count[start_idx: start_idx + self.max_seq_len][::-1])
                        elo_list = self.shuffle(elo_list, elo[start_idx: start_idx + self.max_seq_len][::-1])
                    else:
                        assessmentItemID_list.append(assessmentItemID[start_idx: start_idx + self.max_seq_len][::-1])
                        testId_list.append(testId[start_idx: start_idx + self.max_seq_len][::-1])
                        KnowledgeTag_list.append(KnowledgeTag[start_idx: start_idx + self.max_seq_len][::-1])
                        answerCode_list.append(answerCode[start_idx: start_idx + self.max_seq_len][::-1])
                        #New Feature_list.append(New Feature[start_idx: start_idx + self.max_seq_len][::-1])
                        # userID_list.append([userID]* len(answerCode[::-1]))
                        question_N_list.append(question_N[start_idx: start_idx + self.max_seq_len][::-1])                                        
                        user_correct_answer_list.append(user_correct_answer[start_idx: start_idx + self.max_seq_len][::-1])
                        user_total_answer_list.append(user_total_answer[start_idx: start_idx + self.max_seq_len][::-1])
                        user_acc_list.append(user_acc[start_idx: start_idx + self.max_seq_len][::-1])
                        test_mean_list.append(test_mean[start_idx: start_idx + self.max_seq_len][::-1])
                        test_sum_list.append(test_sum[start_idx: start_idx + self.max_seq_len][::-1])
                        tag_mean_list.append(tag_mean[start_idx: start_idx + self.max_seq_len][::-1])
                        tag_sum_list.append(tag_sum[start_idx: start_idx + self.max_seq_len][::-1])
                        elapsed_list.append(elapsed[start_idx: start_idx + self.max_seq_len][::-1])
                        elapsed_cumsum_list.append(elapsed_cumsum[start_idx: start_idx + self.max_seq_len][::-1])
                        month_list.append(month[start_idx: start_idx + self.max_seq_len][::-1])
                        day_list.append(day[start_idx: start_idx + self.max_seq_len][::-1])
                        hour_list.append(hour[start_idx: start_idx + self.max_seq_len][::-1])
                        dayname_list.append(dayname[start_idx: start_idx + self.max_seq_len][::-1])
                        elapsed_med_list.append(elapsed_med[start_idx: start_idx + self.max_seq_len][::-1])
                        bigclass_list.append(bigclass[start_idx: start_idx + self.max_seq_len][::-1])
                        bigclasstime_list.append(bigclasstime[start_idx: start_idx + self.max_seq_len][::-1])
                        bigclass_acc_list.append(bigclass_acc[start_idx: start_idx + self.max_seq_len][::-1])
                        bigclass_sum_list.append(bigclass_sum[start_idx: start_idx + self.max_seq_len][::-1])
                        bigclass_count_list.append(bigclass_count[start_idx: start_idx + self.max_seq_len][::-1])
                        elo_list.append(elo[start_idx: start_idx + self.max_seq_len][::-1])
                    start_idx += self.window

        ######FE시에 추가해야함
        return assessmentItemID_list, testId_list, KnowledgeTag_list, answerCode_list, question_N_list, user_correct_answer_list, user_total_answer_list, user_acc_list, test_mean_list, test_mean_list,test_sum_list, tag_mean_list, tag_sum_list, elapsed_list,elapsed_cumsum_list, month_list, hour_list, day_list, dayname_list, elapsed_med_list, bigclass_list, bigclasstime_list,bigclass_acc_list,bigclass_sum_list,bigclass_count_list, elo_list

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



