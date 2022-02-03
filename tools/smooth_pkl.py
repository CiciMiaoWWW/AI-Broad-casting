## 清理非主要演员/误检的信息、检测框去抖，同时也解决少数帧没有成功检测的情况
import numpy as np
import pandas as pd
import os
from collections import deque
import argparse
#输入参数
parser = argparse.ArgumentParser()
parser.add_argument('--input_pkl', type = str, default = '/mnt/bd/lwlq-workshop/Person_reID_baseline_pytorch/runs/detect/exp/labels/飞书20220129-210549_labeled.pkl', help = 'input pkl root')
parser.add_argument('--output_pkl', type = str, default = '', help = '输出路径')
parser.add_argument('--frame_win_len', type = int, default = 50, help = '平滑窗口长度')
parser.add_argument('--clean', type = bool, default = True, help = '是否清理非主要演员、可能是误检的信息')
parser.add_argument('--wrong_label', type = int, default = 3, help = '奇怪情境下的label')
parser.add_argument('--actor_num', type = int, default = 3, help = '主要演员个数')
opt = parser.parse_args()

output_pkl = opt.input_pkl[:-4]+'_smooth.pkl'
df=pd.read_pickle(opt.input_pkl)

if opt.clean:
    df = df[(df['actor_id']!=3)]
df = df.reset_index(drop=True)


num = 0
df_new = pd.DataFrame(columns=['frameid','x1','y1','x2','y2','conf','path','actor_id'])
for actor_id in range(opt.actor_num):
    container = deque(maxlen=opt.frame_win_len)
    df_temp = df[(df['actor_id']==actor_id)&(df['frameid']==str(1))]
    # 初始化队列
    for _ in range(opt.frame_win_len):
        container.append([float(df_temp['x1']),float(df_temp['y1']),float(df_temp['x2']),float(df_temp['y2'])])
    df_id_cur = df[(df['actor_id']==actor_id)]
    print(int(df.loc[len(df)-1]['frameid'])+1)
    for i in range(1,int(df.loc[len(df)-1]['frameid'])+1):
        # print(i)
        try:
            df_temp = df[(df['actor_id']==actor_id)&(df['frameid']==str(i))]
            df_temp = df_temp.reset_index(drop=True)
            if len(df_temp)>1:
                df_temp_list[0] = str(i)
            else:
                df_temp_list = list(df_temp.loc[0])
        except:
            df_temp_list[0] = str(i)
        
        # print(df_temp_list)
        container.append([float(df_temp_list[1]),float(df_temp_list[2]),float(df_temp_list[3]),float(df_temp_list[4])])
        # 平均滑窗去抖
        arr_container=np.array(container)
        smooth = np.mean(arr_container,axis=0)
        df_temp_list[1],df_temp_list[2],df_temp_list[3],df_temp_list[4] = smooth[0],smooth[1],smooth[2],smooth[3]
        df_new.loc[num] = df_temp_list
        num += 1
df_new.to_pickle(output_pkl)