def feature_engineering(df):
    df = df.copy()
    
    day_dict = {'Tuesday': 0,
 'Thursday': 1,
 'Monday': 2,
 'Saturday': 3,
 'Friday': 4,
 'Wednesday': 5,
 'Sunday': 6}
    
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    
    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    
    # 문제 푸는데 걸린 시간
    # 10분이상 시간소요는 새로운 문제집을 시작한 것으로 판단
    diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    diff = diff.fillna(pd.Timedelta(seconds=0))
    diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
    df['elapsed'] = diff
    df['elapsed'] = df['elapsed'].apply(lambda x: 0 if x>= 600 else x)
    # 문제 푸는데 걸린 누적 시간
    df['elapsed_cumsum'] = df.groupby('userID')['elapsed'].cumsum()
    #문제 푸는데 걸린 시간의 중앙값
    elapsed_med = df.groupby('userID')['elapsed'].agg(['median'])
    elapsed_med.columns = ['elapsed_med']
    #시간 쪼개기 + 요일
    df['month'] = pd.to_datetime(df.Timestamp).dt.month
    df['day'] = pd.to_datetime(df.Timestamp).dt.day
    df['hour'] = pd.to_datetime(df.Timestamp).dt.hour
    df['dayname'] = pd.to_datetime(df.Timestamp).dt.day_name().map(day_dict)
    
    #대분류/유저
    df['bigclass'] = df['testId'].apply(lambda x : x[2]).astype(int)
    # 유저별 대분류 문제 풀이시간
    bigclasstime = df.groupby(['userID','bigclass']).agg({'elapsed' : 'mean'}).reset_index()

    # 유저별 대분류 문제 횟수
    bigclassCount = df.groupby(['userID','bigclass'])['answerCode'].count().reset_index()
    # 유저별 대분류 문제 정답 횟수
    bigclasssum = df.groupby(['userID','bigclass'])['answerCode'].sum().reset_index()
    v = bigclasssum['answerCode'].values/bigclassCount['answerCode'].values
    bigclasstime['bigclass_acc'] = v
    bigclasstime['bigclass_count']  = bigclassCount['answerCode'].values
    bigclasstime['bigclass_sum'] = bigclasssum['answerCode'].values
    bigclass = bigclasstime.rename(columns = {'elapsed' : 'bigclasstime'})
    df = pd.merge(df,bigclass, on = ['userID','bigclass'],how = 'left')
    
    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'std', 'sum'])
    correct_t.columns = ["test_mean", "test_std", 'test_sum']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'std', 'sum'])
    correct_k.columns = ["tag_mean", 'tag_std', 'tag_sum']

    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, elapsed_med, on =['userID'], how = 'left')
    df.fillna(0,inplace = True)
    return df

#elo함수
def elo(df):
    def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return theta + learning_rate_theta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return beta - learning_rate_beta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def learning_rate_theta(nb_answers):
        return max(0.3 / (1 + 0.01 * nb_answers), 0.04)

    def learning_rate_beta(nb_answers):
        return 1 / (1 + 0.05 * nb_answers)

    def probability_of_good_answer(theta, beta, left_asymptote):
        return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def estimate_parameters(answers_df, granularity_feature_name="assessmentItemID"):
        item_parameters = {
            granularity_feature_value: {"beta": 0, "nb_answers": 0}
            for granularity_feature_value in np.unique(
                answers_df[granularity_feature_name]
            )
        }
        student_parameters = {
            student_id: {"theta": 0, "nb_answers": 0}
            for student_id in np.unique(answers_df.userID)
        }

        print("Parameter estimation is starting...", flush=True)

        for student_id, item_id, left_asymptote, answered_correctly in tqdm(
            zip(
                answers_df.userID.values,
                answers_df[granularity_feature_name].values,
                answers_df.left_asymptote.values,
                answers_df.answerCode.values,
            ),
            total=len(answers_df),
        ):
            theta = student_parameters[student_id]["theta"]
            beta = item_parameters[item_id]["beta"]

            item_parameters[item_id]["beta"] = get_new_beta(
                answered_correctly,
                beta,
                left_asymptote,
                theta,
                item_parameters[item_id]["nb_answers"],
            )
            student_parameters[student_id]["theta"] = get_new_theta(
                answered_correctly,
                beta,
                left_asymptote,
                theta,
                student_parameters[student_id]["nb_answers"],
            )

            item_parameters[item_id]["nb_answers"] += 1
            student_parameters[student_id]["nb_answers"] += 1

        print(f"Theta & beta estimations on {granularity_feature_name} are completed.")
        return student_parameters, item_parameters

    def gou_func(theta, beta):
        return 1 / (1 + np.exp(-(theta - beta)))

    df["left_asymptote"] = 0

    print(f"Dataset of shape {df.shape}")
    print(f"Columns are {list(df.columns)}")

    student_parameters, item_parameters = estimate_parameters(df)

    prob = [
        gou_func(student_parameters[student]["theta"], item_parameters[item]["beta"])
        for student, item in zip(df.userID.values, df.assessmentItemID.values)
    ]

    df["elo"] = prob

    return df