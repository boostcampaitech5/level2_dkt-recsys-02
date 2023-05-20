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
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]
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

            df[col] = test

        def convert_time(s: str):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)
        
        current_dir = os.getcwd()  
        with open('/opt/level2_dkt-recsys-02/code/dkt/models_paramfeature_mapper.pkl', 'wb') as f: 
            pickle.dump(feature_maping_info, f)
        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Fill in if needed
        print('---------Feature Engineering---------')
        # featureEngineering.py를 import 해서 사용
        #df = feature_engineering(df)
        #df = elo(df)
        # 범주형 변수 : [KnowledgeTag,month,day,hour,dayname, bigclass]
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

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        self.user_list = df['userID'].unique().tolist()

        #group = (
        #    df[columns]
        #    .groupby("userID")
        #    .apply(
        #        lambda r: (
        #            r["testId"].values,
        #            r["assessmentItemID"].values,
        #            r["KnowledgeTag"].values,
        #            r["answerCode"].values,
        #        )
        #    )
        #)

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
                    #r[New Feature].values,
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
                    r['day'].values,
                    r['hour'].values,
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
            self.assessmentItemID_list, self.testId_list, self.KnowledgeTag_list, self.answerCode_list = self._data_augmentation()

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
            user_correct_answer = self.user_correct_answer_list[index]
            user_total_answer = self.user_total_answer_list[index]
            user_acc = self.user_acc_list[index]
            test_mean = self.test_mean_list[index]
            test_sum = self.test_sum_list[index]
            tag_mean = self.tag_mean_list[index]
            tag_sum = self.tag_sum_list[index]
            elapsed = self.elapsed_list[index]
            elapsed_cumsum = self.elapsed_cumsum_index[index]
            elasped_med = self.elapsed_med_index[index]
            month = self.month_list[index]
            day = self.day_list[index]
            hour = self.hour_list[index]
            dayname = self.dayname_list[index]
            bigclass = self.bigclass_list[index]
            bigclasstime = self.bigclasstime_list[index]
            bigclass_acc = self.bigclass_acc_list[index]
            bigclass_sum = self.bigclass_sum_list[index]
            bigclass_count = self.bigclass_count_list[index]
            elo = self.elo_list[index]

            if self.args.model == 'lstmtrs':
                now_assessmentItemID = assessmentItemID[1:]
                now_testId = testId[1:]
                now_KnowledgeTag = KnowledgeTag[1:]
                now_answerCode = answerCode[1:]
                
                past_assessmentItemID = assessmentItemID[:-1]
                past_testId = testId[:-1]
                past_KnowledgeTag = KnowledgeTag[:-1]
                past_answerCode = answerCode[:-1]
                data = {
                    "now_testId": torch.tensor(now_testId + 1, dtype=torch.int),
                    "now_assessmentItemID": torch.tensor(now_assessmentItemID + 1, dtype=torch.int),
                    "now_KnowledgeTag": torch.tensor(now_KnowledgeTag + 1, dtype=torch.int),
                    "now_answerCode": torch.tensor(now_answerCode, dtype=torch.int),
                    #"now_New Feature": torch.tensor(now_New Feature, dtype=torch.int),
                    "now_user_correct_answer" : torch.tensor(now_user_correct_answer + 1, dtype=torch.int),
                    "now_user_total_answer" : torch.tensor(now_user_total_answer + 1, dtype=torch.int),
                    "now_user_acc" : torch.tensor(now_user_acc + 1, dtype=torch.int),
                    "now_test_mean" : torch.tensor(now_test_mean + 1, dtype=torch.int),
                    "now_test_sum" : torch.tensor(now_test_sum + 1, dtype=torch.int),
                    "now_tag_mean" : torch.tensor(now_tag_mean + 1, dtype=torch.int),
                    "now_tag_sum" : torch.tensor(now_tag_sum + 1, dtype=torch.int),
                    "now_elapsed" : torch.tensor(now_elapsed + 1, dtype=torch.int),
                    "now_elasped_cumsum" : torch.tensor(now_elapsed_cumsum + 1, dtype=torch.int),
                    "now_elasped_med" : torch.tensor(now_elapsed_med + 1, dtype=torch.int),
                    "now_month" : torch.tensor(now_month + 1, dtype=torch.int),
                    "now_day" : torch.tensor(now_day + 1, dtype=torch.int),
                    "now_hour" : torch.tensor(now_hour + 1, dtype=torch.int),
                    "now_dayname" : torch.tensor(now_dayname + 1, dtype=torch.int),
                    "now_bigclass" : torch.tensor(now_bigclass +1 , dtype=torch.int),
                    "now_bigclass_acc" : torch.tensor(now_bigclass_acc + 1, dtype=torch.int),
                    "now_bigclass_sum" : torch.tensor(now_bigclass_sum + 1, dtype=torch.int),
                    "now_bigclass_count" : torch.tensor(now_bigclass_count + 1, dtype=torch.int),
                    "now_elo" : torch.tensor(now_elo + 1, dtype=torch.int),

                    "past_testId": torch.tensor(past_testId + 1, dtype=torch.int),
                    "past_assessmentItemID": torch.tensor(past_assessmentItemID + 1, dtype=torch.int),
                    "past_KnowledgeTag": torch.tensor(past_KnowledgeTag + 1, dtype=torch.int),
                    "past_answerCode": torch.tensor(past_answerCode, dtype=torch.int),
                    #"past_New Feature": torch.tensor(past_New Feature, dtype=torch.int),
                    "past_user_correct_answer" : torch.tensor(past_user_correct_answer + 1, dtype=torch.int),
                    "past_user_total_answer" : torch.tensor(past_user_total_answer + 1, dtype=torch.int),
                    "past_user_acc" : torch.tensor(past_user_acc + 1, dtype=torch.int),
                    "past_test_mean" : torch.tensor(past_test_mean + 1, dtype=torch.int),
                    "past_test_sum" : torch.tensor(past_test_sum + 1, dtype=torch.int),
                    "past_tag_mean" : torch.tensor(past_tag_mean + 1, dtype=torch.int),
                    "past_tag_sum" : torch.tensor(past_tag_sum + 1, dtype=torch.int),
                    "past_elapsed" : torch.tensor(past_elapsed + 1, dtype=torch.int),
                    "past_elasped_cumsum" : torch.tensor(past_elapsed_cumsum + 1, dtype=torch.int),
                    "past_elasped_med" : torch.tensor(past_elapsed_med + 1, dtype=torch.int),
                    "past_month" : torch.tensor(past_month + 1, dtype=torch.int),
                    "past_day" : torch.tensor(past_day + 1, dtype=torch.int),
                    "past_hour" : torch.tensor(past_hour + 1, dtype=torch.int),
                    "past_dayname" : torch.tensor(past_dayname + 1, dtype=torch.int),
                    "past_bigclass" : torch.tensor(past_bigclass + 1, dtype=torch.int),
                    "past_bigclass_acc" : torch.tensor(past_bigclass_acc + 1, dtype=torch.int),
                    "past_bigclass_sum" : torch.tensor(past_bigclass_sum + 1, dtype=torch.int),
                    "past_bigclass_count" : torch.tensor(past_bigclass_count + 1, dtype=torch.int),
                    "past_elo" : torch.tensor(past_elo + 1, dtype=torch.int)
                }
                seq_len = len(now_answerCode)
                return data
            
            else:
                data = {
                "testId": torch.tensor(testId + 1, dtype=torch.int),
                "assessmentItemID": torch.tensor(assessmentItemID + 1, dtype=torch.int),
                "KnowledgeTag": torch.tensor(KnowledgeTag + 1, dtype=torch.int),
                "answerCode": torch.tensor(answerCode, dtype=torch.int),
                #"userID" : torch.tensor(userID, dtype=torch.int),
                #New Feature = torch.tensor(New Feature + 1, dtype=torch.int)
                "user_correct_answer" : torch.tensor(user_correct_answer + 1, dtype=torch.int),
                "user_total_answer" : torch.tensor(user_total_answer + 1, dtype=torch.int),
                "user_acc" : torch.tensor(user_acc + 1, dtype=torch.int),
                "test_mean" : torch.tensor(test_mean + 1, dtype=torch.int),
                "test_sum" : torch.tensor(test_sum + 1, dtype=torch.int),
                "tag_mean" : torch.tensor(tag_mean + 1, dtype=torch.int),
                "tag_sum" : torch.tensor(tag_sum + 1, dtype=torch.int),
                "elapsed" : torch.tensor(elapsed + 1, dtype=torch.int),
                "elapsed_cumsum" : torch.tensor(elapsed_cumsum + 1, dtype=torch.int),
                "elasped_med" : torch.tensor(elapsed_med + 1, dtype=torch.int),
                "month" : torch.tensor(month + 1, dtype=torch.int),
                "day" : torch.tensor(day + 1, dtype=torch.int),
                "hour" : torch.tensor(hour + 1, dtype=torch.int),
                "dayname" : torch.tensor(dayname + 1, dtype=torch.int),
                "bigclass" : torch.tensor(bigclass + 1, dtype=torch.int),
                "bigclass_acc" : torch.tensor(bigclass_acc + 1, dtype=torch.int),
                "bigclass_sum" : torch.tensor(bigclass_sum + 1, dtype=torch.int),
                "bigclass_count" : torch.tensor(bigclass_count + 1, dtype=torch.int),
                "elo" : torch.tensor(elo + 1, dtype=torch.int)
                }
                seq_len = len(answerCode)


####################Mask 만들기
            if seq_len >= self.max_seq_len:
                mask = torch.ones(self.max_seq_len, dtype=torch.int16)
            else:
                for feature in data:
                    # Pre-padding non-valid sequences
                    tmp = torch.zeros(self.max_seq_len)
                    tmp[self.max_seq_len-seq_len:] = data[feature]
                    data[feature] = tmp
                mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
                mask[-seq_len:] = 1

            data["mask"] = mask
####################Sliding Window 미적용 시
        else:
            row = self.grouped_value[index]
####################FE 추가 시 추가해야함
            testId, assessmentItemID, KnowledgeTag, answerCode = row[0], row[1], row[2], row[3]
            data = {
                "testId": torch.tensor(testId + 1, dtype=torch.int),
                "assessmentItemID": torch.tensor(assessmentItemID + 1, dtype=torch.int),
                "KnowledgeTag": torch.tensor(KnowledgeTag + 1, dtype=torch.int),
                "answerCode": torch.tensor(answerCode, dtype=torch.int),
                #"userID" : torch.tensor(userID, dtype=torch.int),
                #New Feature = torch.tensor(New Feature + 1, dtype=torch.int)
                "user_correct_answer" : torch.tensor(user_correct_answer + 1, dtype=torch.int),
                "user_total_answer" : torch.tensor(user_total_answer + 1, dtype=torch.int),
                "user_acc" : torch.tensor(user_acc + 1, dtype=torch.int),
                "test_mean" : torch.tensor(test_mean + 1, dtype=torch.int),
                "test_sum" : torch.tensor(test_sum + 1, dtype=torch.int),
                "tag_mean" : torch.tensor(tag_mean + 1, dtype=torch.int),
                "tag_sum" : torch.tensor(tag_sum + 1, dtype=torch.int),
                "elapsed" : torch.tensor(elapsed + 1, dtype=torch.int),
                "elapsed_cumsum" : torch.tensor(elapsed_cumsum + 1, dtype=torch.int),
                "elasped_med" : torch.tensor(elapsed_med + 1, dtype=torch.int),
                "month" : torch.tensor(month + 1, dtype=torch.int),
                "day" : torch.tensor(day + 1, dtype=torch.int),
                "hour" : torch.tensor(hour + 1, dtype=torch.int),
                "dayname" : torch.tensor(dayname + 1, dtype=torch.int),
                "bigclass" : torch.tensor(bigclass + 1, dtype=torch.int),
                "bigclass_acc" : torch.tensor(bigclass_acc + 1, dtype=torch.int),
                "bigclass_sum" : torch.tensor(bigclass_sum + 1, dtype=torch.int),
                "bigclass_count" : torch.tensor(bigclass_count + 1, dtype=torch.int),
                "elo" : torch.tensor(elo + 1, dtype=torch.int)
                }
            
####################Mask 만들기       
            seq_len = len(answerCode)

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
                    #userID_list.append([userID]* len(answerCode[::-1
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
                    #userID_list.append([userID]* len(answerCode[::-1]))
                    user_correct_answer_list = user_correct_answer[start_idx: start_idx + self.max_seq_len][::-1]
user_total_answer_list = user_total_answer[start_idx: start_idx + self.max_seq_len][::-1]
user_acc_list = user_acc[start_idx: start_idx + self.max_seq_len][::-1]
test_mean_list = test_mean[start_idx: start_idx + self.max_seq_len][::-1]
test_sum_list = test_sum[start_idx: start_idx + self.max_seq_len][::-1]
tag_mean_list = tag_mean[start_idx: start_idx + self.max_seq_len][::-1]
tag_sum_list = tag_sum[start_idx: start_idx + self.max_seq_len][::-1]
elapsed_list = elapsed[start_idx: start_idx + self.max_seq_len][::-1]
elapsed_cumsum_list = elapsed_cumsum[start_idx: start_idx + self.max_seq_len][::-1]
month_list = month[start_idx: start_idx + self.max_seq_len][::-1]
day_list = day[start_idx: start_idx + self.max_seq_len][::-1]
hour_list = hour[start_idx: start_idx + self.max_seq_len][::-1]
dayname_list = dayname[start_idx: start_idx + self.max_seq_len][::-1]
elapsed_med_list = elapsed_med[start_idx: start_idx + self.max_seq_len][::-1]
bigclass_list = bigclass[start_idx: start_idx + self.max_seq_len][::-1]
bigclasstime_list = bigclasstime[start_idx: start_idx + self.max_seq_len][::-1]
bigclass_acc_list = bigclass_acc[start_idx: start_idx + self.max_seq_len][::-1]
bigclass_sum_list = bigclass_sum[start_idx: start_idx + self.max_seq_len][::-1]
bigclass_count_list = bigclass_count[start_idx: start_idx + self.max_seq_len][::-1]
elo_list = elo[start_idx: start_idx + self.max_seq_len][::-1]
                    start_idx += self.window

        ######FE시에 추가해야함
        return assessmentItemID_list, testId_list, KnowledgeTag_list, answerCode_list user_correct_answer_list, user_total_answer_list, user_acc_list, test_mean_list, test_sum_list, tag_mean_list, tag_sum_list, elapsed_list, elapsed_cumsum_list, month_list, day_list, hour_list, dayname_list, elapsed_med_list, bigclass_list, bigclasstime_list, bigclass_acc_list, bigclass_sum_list, bigclass_count_list, elo_list #New Feature_list

def get_loaders(args, train: np.ndarray, valid: np.ndarray) -> Tuple[torch.utils.data.DataLoader]:
    pin_memory = False
    train_loader, valid_loader = None, None

    if args.model == 'lstmtrs':
        if train is not None:
            trainset = DKTDataset(train, args)
            train_loader = torch.utils.data.DataLoader(
                trainset,
                num_workers=args.num_workers,
                shuffle=True,
                batch_size=args.batch_size,
                pin_memory=pin_memory,
                collate_fn=sequence_collate,
            )
        if valid is not None:
            valset = DKTDataset(valid, args)
            valid_loader = torch.utils.data.DataLoader(
                valset,
                num_workers=args.num_workers,
                shuffle=False,
                batch_size=args.batch_size,
                pin_memory=pin_memory,
                collate_fn=sequence_collate,
            )
    else:
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

def pad_sequence(seq, max_len, padding_value = 0):
    try:
        seq_len, col = seq.shape
        padding = np.zeros((max_len - seq_len, col)) + padding_value
    except:
        seq_len = seq.shape[0]
        padding = np.zeros((max_len - seq_len, )) + padding_value

    padding_seq = np.concatenate([padding, seq])

    return padding_seq


def sequence_collate(samples):
    max_len = 0
    for sample in samples:
        seq_len = len(sample['past_testId'])
        if max_len < seq_len:
            max_len = seq_len

    now_assessmentItemID = []
    now_testId = []
    now_KnowledgeTag = []
    now_answerCode = []
    now_user_correct_answer = []
    now_user_total_answer = []
    now_user_acc = []
    now_test_mean = []
    now_test_sum = []
    now_tag_mean = []
    now_tag_sum = []
    now_elapsed = []
    now_elapsed_cumsum = []
    now_month = []
    now_day = []
    now_hour = []
    now_dayname = []
    now_elapsed_med = []
    now_bigclass = []
    now_bigclasstime = []
    now_bigclass_acc = []
    now_bigclass_sum = []
    now_bigclass_count = []
    now_elo = []

    past_assessmentItemID = []
    past_testId = []
    past_KnowledgeTag = []
    past_answerCode = []
    past_user_correct_answer = []
    past_user_total_answer = []
    past_user_acc = []
    past_test_mean = []
    past_test_sum = []
    past_tag_mean = []
    past_tag_sum = []
    past_elapsed = []
    past_elapsed_cumsum = []
    past_month = []
    past_day = []
    past_hour = []
    past_dayname = []
    past_elapsed_med = []
    past_bigclass = []
    past_bigclasstime = []
    past_bigclass_acc = []
    past_bigclass_sum = []
    past_bigclass_count = []
    past_elo = []

    for sample in samples:
        now_assessmentItemID += [pad_sequence(sample['now_assessmentItemID'] + 1, max_len = max_len, padding_value = 0)]
        now_testId += [pad_sequence(sample['now_testId'] + 1, max_len = max_len, padding_value = 0)]
        now_KnowledgeTag += [pad_sequence(sample['now_KnowledgeTag'] + 1, max_len = max_len, padding_value = 0)]
        now_answerCode += [pad_sequence(sample['now_answerCode'], max_len = max_len, padding_value = -1)]
        now_user_correct_answer += [pad_sequence(sample['now_user_correct_answer'] + 1, max_len=max_len, padding_value=0)]
        now_user_total_answer += [pad_sequence(sample['now_user_total_answer'] + 1, max_len=max_len, padding_value=0)]
        now_user_acc += [pad_sequence(sample['now_user_acc'] + 1, max_len=max_len, padding_value=0)]
        now_test_mean += [pad_sequence(sample['now_test_mean'] + 1, max_len=max_len, padding_value=0)]
        now_test_sum += [pad_sequence(sample['now_test_sum'] + 1, max_len=max_len, padding_value=0)]
        now_tag_mean += [pad_sequence(sample['now_tag_mean'] + 1, max_len=max_len, padding_value=0)]
        now_tag_sum += [pad_sequence(sample['now_tag_sum'] + 1, max_len=max_len, padding_value=0)]
        now_elapsed += [pad_sequence(sample['now_elapsed'] + 1, max_len=max_len, padding_value=0)]
        now_elapsed_cumsum += [pad_sequence(sample['now_elapsed_cumsum'] + 1, max_len=max_len, padding_value=0)]
        now_month += [pad_sequence(sample['now_month'] + 1, max_len=max_len, padding_value=0)]
        now_day += [pad_sequence(sample['now_day'] + 1, max_len=max_len, padding_value=0)]
        now_hour += [pad_sequence(sample['now_hour'] + 1, max_len=max_len, padding_value=0)]
        now_dayname += [pad_sequence(sample['now_dayname'] + 1, max_len=max_len, padding_value=0)]
        now_elapsed_med += [pad_sequence(sample['now_elapsed_med'] + 1, max_len=max_len, padding_value=0)]
        now_bigclass += [pad_sequence(sample['now_bigclass'] + 1, max_len=max_len, padding_value=0)]
        now_bigclasstime += [pad_sequence(sample['now_bigclasstime'] + 1, max_len=max_len, padding_value=0)]
        now_bigclass_acc += [pad_sequence(sample['now_bigclass_acc'] + 1, max_len=max_len, padding_value=0)]
        now_bigclass_sum += [pad_sequence(sample['now_bigclass_sum'] + 1, max_len=max_len, padding_value=0)]
        now_bigclass_count += [pad_sequence(sample['now_bigclass_count'] + 1, max_len=max_len, padding_value=0)]
        now_elo += [pad_sequence(sample['now_elo'] + 1, max_len=max_len, padding_value=0)]


        past_assessmentItemID += [pad_sequence(sample['past_assessmentItemID'] + 1, max_len = max_len, padding_value = 0)]
        past_testId += [pad_sequence(sample['past_testId'] + 1, max_len = max_len, padding_value = 0)]
        past_KnowledgeTag += [pad_sequence(sample['past_KnowledgeTag'] + 1, max_len = max_len, padding_value = 0)]
        past_answerCode += [pad_sequence(sample['past_answerCode'] + 1, max_len = max_len, padding_value = 0)]    
        past_user_correct_answer += [pad_sequence(sample['past_user_correct_answer'] + 1, max_len=max_len, padding_value=0)]
        past_user_total_answer += [pad_sequence(sample['past_user_total_answer'] + 1, max_len=max_len, padding_value=0)]
        past_user_acc += [pad_sequence(sample['past_user_acc'] + 1, max_len=max_len, padding_value=0)]
        past_test_mean += [pad_sequence(sample['past_test_mean'] + 1, max_len=max_len, padding_value=0)]
        past_test_sum += [pad_sequence(sample['past_test_sum'] + 1, max_len=max_len, padding_value=0)]
        past_tag_mean += [pad_sequence(sample['past_tag_mean'] + 1, max_len=max_len, padding_value=0)]
        past_tag_sum += [pad_sequence(sample['past_tag_sum'] + 1, max_len=max_len, padding_value=0)]
        past_elapsed += [pad_sequence(sample['past_elapsed'] + 1, max_len=max_len, padding_value=0)]
        past_elapsed_cumsum += [pad_sequence(sample['past_elapsed_cumsum'] + 1, max_len=max_len, padding_value=0)]
        past_month += [pad_sequence(sample['past_month'] + 1, max_len=max_len, padding_value=0)]
        past_day += [pad_sequence(sample['past_day'] + 1, max_len=max_len, padding_value=0)]
        past_hour += [pad_sequence(sample['past_hour'] + 1, max_len=max_len, padding_value=0)]
        past_dayname += [pad_sequence(sample['past_dayname'] + 1, max_len=max_len, padding_value=0)]
        past_elapsed_med += [pad_sequence(sample['past_elapsed_med'] + 1, max_len=max_len, padding_value=0)]
        past_bigclass += [pad_sequence(sample['past_bigclass'] + 1, max_len=max_len, padding_value=0)]
        past_bigclasstime += [pad_sequence(sample['past_bigclasstime'] + 1, max_len=max_len, padding_value=0)]
        past_bigclass_acc += [pad_sequence(sample['past_bigclass_acc'] + 1, max_len=max_len, padding_value=0)]
        past_bigclass_sum += [pad_sequence(sample['past_bigclass_sum'] + 1, max_len=max_len, padding_value=0)]
        past_bigclass_count += [pad_sequence(sample['past_bigclass_count'] + 1, max_len=max_len, padding_value=0)]
        past_elo += [pad_sequence(sample['past_elo'] + 1, max_len=max_len, padding_value=0)]
    return {
            "now_testId": torch.tensor(now_testId, dtype=torch.int),
            "now_assessmentItemID": torch.tensor(now_assessmentItemID, dtype=torch.int),
            "now_KnowledgeTag": torch.tensor(now_KnowledgeTag, dtype=torch.int),
            "now_answerCode": torch.tensor(now_answerCode, dtype=torch.int),
            #"now_New Feature": torch.tensor(now_New Feature, dtype=torch.int),
            "now_user_correct_answer": torch.tensor(now_user_correct_answer, dtype=torch.int),
            "now_user_total_answer": torch.tensor(now_user_total_answer, dtype=torch.int),
            "now_user_acc": torch.tensor(now_user_acc, dtype=torch.float),
            "now_test_mean": torch.tensor(now_test_mean, dtype=torch.float),
            "now_test_sum": torch.tensor(now_test_sum, dtype=torch.float),
            "now_tag_mean": torch.tensor(now_tag_mean, dtype=torch.float),
            "now_tag_sum": torch.tensor(now_tag_sum, dtype=torch.float),
            "now_elapsed": torch.tensor(now_elapsed, dtype=torch.float),
            "now_elapsed_cumsum": torch.tensor(now_elapsed_cumsum, dtype=torch.float),
            "now_month": torch.tensor(now_month, dtype=torch.int),
            "now_day": torch.tensor(now_day, dtype=torch.int),
            "now_hour": torch.tensor(now_hour, dtype=torch.int),
            "now_dayname": torch.tensor(now_dayname, dtype=torch.int),
            "now_elapsed_med": torch.tensor(now_elapsed_med, dtype=torch.float),
            "now_bigclass": torch.tensor(now_bigclass, dtype=torch.int),
            "now_bigclasstime": torch.tensor(now_bigclasstime, dtype=torch.int),
            "now_bigclass_acc": torch.tensor(now_bigclass_acc, dtype=torch.float),
            "now_bigclass_sum": torch.tensor(now_bigclass_sum, dtype=torch.int),
            "now_bigclass_count": torch.tensor(now_bigclass_count, dtype=torch.int),
            "now_elo": torch.tensor(now_elo, dtype=torch.float)
        
        
            "past_testId": torch.tensor(past_testId, dtype=torch.int),
            "past_assessmentItemID": torch.tensor(past_assessmentItemID, dtype=torch.int),
            "past_KnowledgeTag": torch.tensor(past_KnowledgeTag, dtype=torch.int),
            "past_answerCode": torch.tensor(past_answerCode, dtype=torch.int),
            #"past_New Feature": torch.tensor(past_New Feature, dtype=torch.int),
            "past_user_correct_answer": torch.tensor(past_user_correct_answer, dtype=torch.int),
            "past_user_total_answer": torch.tensor(past_user_total_answer, dtype=torch.int),
            "past_user_acc": torch.tensor(past_user_acc, dtype=torch.float),
            "past_test_mean": torch.tensor(past_test_mean, dtype=torch.float),
            "past_test_sum": torch.tensor(past_test_sum, dtype=torch.float),
            "past_tag_mean": torch.tensor(past_tag_mean, dtype=torch.float),
            "past_tag_sum": torch.tensor(past_tag_sum, dtype=torch.float),
            "past_elapsed": torch.tensor(past_elapsed, dtype=torch.float),
            "past_elapsed_cumsum": torch.tensor(past_elapsed_cumsum, dtype=torch.float),
            "past_month": torch.tensor(past_month, dtype=torch.int),
            "past_day": torch.tensor(past_day, dtype=torch.int),
            "past_hour": torch.tensor(past_hour, dtype=torch.int),
            "past_dayname": torch.tensor(past_dayname, dtype=torch.int),
            "past_elapsed_med": torch.tensor(past_elapsed_med, dtype=torch.float),
            "past_bigclass": torch.tensor(past_bigclass, dtype=torch.int),
            "past_bigclasstime": torch.tensor(past_bigclasstime, dtype=torch.int),
            "past_bigclass_acc": torch.tensor(past_bigclass_acc, dtype=torch.float),
            "past_bigclass_sum": torch.tensor(past_bigclass_sum, dtype=torch.int),
            "past_bigclass_count": torch.tensor(past_bigclass_count, dtype=torch.int),
            "past_elo": torch.tensor(past_elo, dtype=torch.float)
            }