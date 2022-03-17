import numpy as np
import scipy.signal 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, default='output.txt',
	help="input file")
ap.add_argument("-p", "--people", default=3,
	help="number of people detected")
ap.add_argument("-m", "--mouth-ar-thresh", required=False, default=0.6,
    help="threshold of open-moth detection")
args = vars(ap.parse_args())

MOUTH_AR_THRESH = float(args["mouth_ar_thresh"])

people = int(args['people']) 

info = [[] for i in range(people)]

# input
pos = []
v = []
with open(args["input"], "r") as f:
    for line in f.readlines():
        line = line.split(" ")
        # print(line)
        pos.append([
            int(line[1].split("(")[1].split(",")[0]), 
            int(line[2].split(")")[0])
        ])
        v.append({
            "frame" : int(line[0]),
            "e" : float(line[-2]),
            "v" : int(line[-1].split("\n")[0])
        })

# group

size = len(pos)
# BUG：就是第一帧必须检测到所有人的嘴，不然可能会挂
last = [[] for i in range(people)]
for i in range(people):
    last[i] = pos[i]
    info[i].append(v[i])

cnt = 1
now = []
for i in range(people, size):
    if cnt != v[i]['frame']:
        cnt = cnt + 1
        now = []
    now.append([pos[i], v[i]])
    if i == size - 1 or v[i + 1]['frame'] != v[i]['frame']:
        for ele in now:
            dis = 1e99
            p = -1
            # print("ele = ", ele)
            for j in range(people):
                if (last[j][0] - ele[0][0])**2 + (last[j][1] - ele[0][1])**2 < dis:
                    dis = (last[j][0] - ele[0][0])**2 + (last[j][1] - ele[0][1])**2
                    p = j
            # print("j = ", pos)
            info[p].append(ele[1])
            last[p] = ele[0]

cnt = cnt + 1
old_info = info
info = [[[] for j in range(cnt)] for i in range(people)]
for people_id in range(len(old_info)):
    for ele in old_info[people_id]:
        info[people_id][ele['frame']] = {
            'v' : ele['v'],
            'e' : ele['e']
        } 
    for i in range(cnt):
        if info[people_id][i] == []:
            info[people_id][i] = {
                'v' : 0,
                'e' : 0.
            }


# smooth
def smooth(data):
    '''
        Savitzky-Golay
    '''
    # return data
    return scipy.signal.savgol_filter(data,30,4)

old_info = info
info = []
for people_info in old_info:
    data = [i['e'] for i in people_info]
    # print(smooth(data))
    tmp_list = []
    tmp_data = smooth(data)
    for i in range(len(tmp_data)):
        if tmp_data[i] > MOUTH_AR_THRESH:
            tmp_list.append([i, 1])
        else:
            tmp_list.append([i, 0])
    info.append(tmp_list)


# output

def getInterval(data):
    ans = []
    lastl = -1 
    ele_filtered = filter(lambda x: x[1] == 1, data)
    lastr = None
    for i in ele_filtered:
        if lastr != i[0] - 1:
            if lastr != None:
                ans.append((lastl + 1, lastr))
            lastl = i[0] - 1
        lastr = i[0]
    return ans

id = 0
for ele in info:
    ans = getInterval(ele)
    print(id, ans)
    id = id + 1





            
            

