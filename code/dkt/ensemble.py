import pandas as pd
import numpy as np
import argparse 
import os
from pytz import timezone
from datetime import datetime
import pdb

class Ensemble:
    def __init__(self, filenames:str, filepath:str):
    # filenames : [bert_submission,lstm_submission] ,
    # filepath : './outputs/'
        self.filepath = filepath
        self.filenames = filenames
        self.output_list = []

        output_path = [filepath + filename for filename in self.filenames] 
        self.output_frame = pd.read_csv(output_path[0]).drop("prediction",axis = 1)
        self.output_df  = self.output_frame.copy()
        self.csv_nums = len(output_path)

        for path in output_path:
            self.output_list.append(pd.read_csv(path)["prediction"].to_list())
        for filename,output in zip(self.filenames,self.output_list):
            self.output_df[filename] = output
        
    def simple_weighted(self,weight: list):
        """
        Ensembles with manually designated weight
        """
        if not len(self.output_list) == len(weight):
            raise ValueError("model과 weight의 길이가 일치하지 않습니다.")
        if np.sum(weight) != 1:
            raise ValueError("weight의 합이 1이 되게 조정해주세요.")
        
        pred_arr = np.append([self.output_list[0]],[self.output_list[1]],axis=0)
        for i in range(2,len(self.output_list)):
            pred_arr = np.append(pred_arr,self.output_list[i],axis=0)
        result = np.dot(pred_arr.T,np.array(weight))
        return result.tolist()
    
    def average_weighted(self):
        weight = [1/len(self.output_list) for _ in range(len(self.output_list))]
        pred_weight_list = [
                pred * np.array(w) for pred,w in zip(self.output_list,weight)
        ]
        result = np.sum(pred_weight_list, axis=0)
        return result.tolist()
    
    def mixed(self):
        """

        """
        result = self.output_df[self.filenames[0]].copy()
        for idx in range(len(self.filenames)-1):
            pre_idx = self.filenames[idx]
            post_idx
    

def main(args):
    # --ensemble_files bert_submission,lstm_submission -> [bert_submission,lstm_submission]
    #file_list = sum(args.ensemble_files,[])

    file_list = [item for item in args.ensemble_files.split(',')]  #[bert_submission,lstm_submission]
    print(file_list)
    if len(file_list) <2:
        raise ValueError("Model을 적어도 2개는 입력해 주세요.")

    filepath = args.item_path
    savepath = args.result_path

    os.makedirs(filepath, exist_ok=True)
    if os.listdir(filepath) == []:
        raise ValueError(f"Put inference.csv files in folder path {filepath}")
    os.makedirs(savepath, exist_ok=True)
    
    en = Ensemble(file_list, args.item_path) # file_list : [bert_submission,lstm_submission] , args.item_path : './outputs/'
    
    if args.strategy == "weighted":
        if args.ensemble_weight:
            strategy_title = "weighted-"+'-'.join(map(str,*args.ensemble_weight))
            result = en.single_weighted(*args.ensemble_weight)
        else:
            strategy_title == "average weighted"
            result = en.average_weighted(*args.ensemble_weight)
    elif args.strategy == "mixed":
        strategy_title = args.strategy
        result = en.mixed()
    elif args.strategy == "hardsoft":
        strategy_title = args.strategy
        result = en.hardsoft()
    else:
        pass

    en.output_frame['prediction'] = result 
    output = en.output_frame.copy()
    now = datetime.now(timezone(Asia/Seoul)).strftime(f"%Y-%m-%d_%H:%M")

    output.to_csv(f'{savepath}{now}-{strategy_title}.csv',index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    arg("--ensemble_files",type=str, default=None,
        help='required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. 이 때, .csv와 같은 확장자는 입력하지 않습니다.')
    arg('--result_path',type=str, default='./outputs/ensemble_list/',
        help='optional: 앙상블한 파일을 저장하는 경로를 전달합니다. (default:"./outputs/ensemble_list/")')
    arg('--item_path',type=str, default='./outputs/',
        help='optional: 앙상블할 파일을 가져오는 경로를 전달합니다. (default:"./outputs/")')
    arg('--result_name',type=str, default='ensemble',
        help='optional: 앙상블한 파일을 저장할 이름을 전달합니다. (default:"ensemble")')
    arg('--strategy',type=str, default='weighted',
        choices= ['weighted','mixed','hardsoft'],
        help='optional: ['weighted','mixed','hardsoft'] 중 앙상블 전략을 선택합니다. (default:"weighted")')
    arg('--ensemble_weight', nargs='+',default=None,
        type=lambda s: [float(item) for item in s.split(',')],
        help='optional: Weighted 앙상블에서 각 모델별 가중치를 조정할 수 있습니다.(default:None, average로 결과 출력 )')
    
    

    args = parser.parse_args()
    main(args)