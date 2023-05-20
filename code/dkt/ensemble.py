import pandas as pd
import numpy as np
import argparse 
import pdb

def Ensemble(filenames:str, filepath:str):
    # filenames : [bert_submission,lstm_submission] ,
    # filepath : './outputs/'
    num_model = len(filenames) # 2

    load_list = pd.read_csv(filepath+filenames[0]+".csv")
    load_list = load_list['prediction']

    for i in range(1,num_model): #i = 0
        filename = filenames[i] # filename = bert_submission
        load_result= pd.read_csv(filepath+filename+".csv")  #'./outputs/'+ 'bert_submission'
        load_list+= load_result['prediction']

    en = pd.read_csv(filepath+filenames[0]+".csv") # ensemble 모델의 파일은 마지막 모델의 파일을 복사하여 생성
    
    en['prediction'] = load_list / num_model
    return en


def main(args):
    # --ensemble_files bert_submission,lstm_submission -> [bert_submission,lstm_submission]
    #file_list = sum(args.ensemble_files,[])
    
    file_list = [item for item in args.ensemble_files.split(',')]  #[bert_submission,lstm_submission]
    print(file_list)
    if len(file_list) <2:
        raise ValueError("Model을 적어도 2개는 입력해 주세요.")

    en = Ensemble(file_list, args.ensemble_item_path) # file_list : [bert_submission,lstm_submission] , args.ensemble_item_path : './outputs/'
    return en.to_csv(args.ensemble_result_path+args.ensemble_name+'.csv',index=False) #'./outputs/ensemble_list/'+'ensemble'+ '.csv'

pdb.set_trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    arg("--ensemble_files",type=str, default=None,
        help='required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. 이 때, .csv와 같은 확장자는 입력하지 않습니다.')
    arg('--ensemble_result_path',type=str, default='./outputs/ensemble_list/',
        help='optional: 앙상블한 파일을 저장하는 경로를 전달합니다. (default:"./outputs/ensemble_list/")')
    arg('--ensemble_item_path',type=str, default='./outputs/',
        help='optional: 앙상블할 파일을 가져오는 경로를 전달합니다. (default:"./outputs/")')
    arg('--ensemble_name',type=str, default='ensemble',
        help='optional: 앙상블한 파일을 저장할 이름을 전달합니다. (default:"ensemble")')
    
    # arg('--ensemble_strategy', type=str, default='weighted',
    #     choices=['weighted','mixed'],
    #     help='optional: [mixed, weighted] 중 앙상블 전략을 선택해 주세요. (default="weighted")')
    # arg('--ensemble_weight', nargs='+',default=None,
    #     type=lambda s: [float(item) for item in s.split(',')],
    #     help='optional: weighted 앙상블 전략에서 각 결과값의 가중치를 조정할 수 있습니다.')
    

    args = parser.parse_args()
    main(args)