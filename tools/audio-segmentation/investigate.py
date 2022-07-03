import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import argparse
from pathlib import Path
from os.path import join
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
import os
import pandas as pd


parser = argparse.ArgumentParser(description="testing and prepare npy for plot")
parser.add_argument("--filename", type=str, default="result_string", help="filename, default is result.")
parser.add_argument("--channel", type=str, default="channel_2", help="channel_1/channel_2")
parser.add_argument("--outputfilename", type=int, default="0", help="output file name for current video chunck")


args_opt = parser.parse_args()

def similarity(a,b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

filename= args_opt.filename
channel= args_opt.channel # by default channel2
with open('{}.npy'.format(filename), 'rb') as f:
    a = np.load(f)

print(a.shape)
# focus on track 1

if channel=='channel_2':
    data_to_plot_list = a[:, :, 1:, :]
else:
    data_to_plot_list=a[:,:,:1,:]
print(data_to_plot_list.shape)

modified_output=[]

for i,file in enumerate(data_to_plot_list):
    #squueze
    data_to_plot=data_to_plot_list[i].squeeze()

    print(data_to_plot.shape)

    # cosin similarity
    data_to_plot=list(data_to_plot)
    a_list = data_to_plot[:-1]
    b_list = data_to_plot[1:]

    print(len(a_list))
    print(len(b_list))

    cos_similarity_result=[]

    for j in range(len(a_list)):
        cos_similarity_result.append(similarity(a_list[j],b_list[j]))

    print(len(cos_similarity_result))
    cos_similarity_result = np.array(cos_similarity_result)
    print(cos_similarity_result.shape)

    data_to_plot=np.array(data_to_plot)

    #transpose
    timestamp_diff=data_to_plot=data_to_plot.transpose(1,0)
    print(data_to_plot.shape)

    # # time stamp
    # timestamp_diff = np.diff(timestamp_diff, axis=-1)
    # print(timestamp_diff.shape)
    #
    # # sum at axis 0
    # timestamp_diff=data_to_plot.sum(axis=0)
    # print(timestamp_diff.shape)





    # time stamp percentile
    q3, q2, q1 = np.percentile(cos_similarity_result, [98, 50, 2])
    iqr = q3 - q1

    print(q1, q2, q3, iqr)

    # get the result
    output_to_npy = np.array(np.where(cos_similarity_result<q1)).squeeze()+1

    #eliminate consubsqeunt frames
    diff = np.diff(output_to_npy) #find the
    print(output_to_npy)
    modified_array = np.delete(output_to_npy, np.where(diff<=20))
    print(modified_array)
    # to time stamp
    modified_array= modified_array/600*60
    offset=(i*60)%120
    modified_array= modified_array+offset

    modified_output.append(modified_array)



    # print(timestamp_diff[0])
    # print(timestamp_diff[1])


    r=data_to_plot
    #print(center)
    #r = normalized = np.where(r>0,r/r.max(),np.where(r<0,-r/r.min(),r))

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    plt.title('{}_{}_{}_heatmap'.format(filename,i+1,channel))
    ax = sns.heatmap(r,center=0,ax=ax)

    dir = Path(".\\investigation_result")
    os.makedirs(dir, exist_ok=True)
    # save the figure
    plt.savefig('investigation_result/{}_{}_{}_plot.png'.format(filename,i+1,channel), dpi=300, bbox_inches='tight')
    #plt.show()

#wrte it outside of loop
print('saving final timestamp results')
print('each array correspond to one minute interval')
print(modified_output)
dir= Path(".\\timestamp_results")
os.makedirs(dir, exist_ok = True)
output_file_name = 'timestamp_{}_{}_v2.pkl'.format(args_opt.filename,args_opt.outputfilename)

output_file_dir = join(dir, output_file_name)
# wriet the test results into a pickle file


df = pd.DataFrame(columns=['t1','t2'])

firstmin = modified_output[0]
secondmin = modified_output[1]

time_offset = 0

new_ts = []
for t in firstmin:
    if t > time_offset:
        new_ts.append(t - time_offset)
for t in secondmin:
    if t > time_offset:
        new_ts.append(t - time_offset)

for num, ts in enumerate(new_ts):
    if num < len(new_ts) - 1:
        df.loc[num] = [new_ts[num], new_ts[num + 1]]

print(df)

df.to_pickle(output_file_dir)

