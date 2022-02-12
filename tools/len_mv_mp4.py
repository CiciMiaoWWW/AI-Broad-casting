import cv2, os
from moviepy import *
from moviepy.editor import *
import numpy as np
import imageio
from tqdm import tqdm
import pandas as pd

class MvLens:
    def __init__(self, fullwidth, fullheight, step=300, padding_scale=1.1):
        self.step = step
        self.fullwidth = fullwidth
        self.fullheight = fullheight
        self.padding_scale = padding_scale

    def apply_lensmv(self, x1, y1, x2, y2, curr_step, mvtint):
        #0 None, 1 left, -1 right, 2 up, -2 down, 3 zoomin, -3 zoomout, 4 left zoomin, -4 right zoomout, 5 right zoomin, -5 left zoomout, 6 up zoomin, -6 down zoomout, 7 down zoomin, -7 up zoomout
        if mvtint == 1:
            x1_out, y1_out, x2_out, y2_out = self.apply_left(x1, y1, x2, y2, curr_step)
        elif mvtint == -1:
            x1_out, y1_out, x2_out, y2_out = self.apply_right(x1, y1, x2, y2, curr_step)
        elif mvtint == 2:
            x1_out, y1_out, x2_out, y2_out = self.apply_right(x1, y1, x2, y2, curr_step)
        elif mvtint == -2:
            x1_out, y1_out, x2_out, y2_out = self.apply_down(x1, y1, x2, y2, curr_step)
        elif mvtint == 3:
            x1_out, y1_out, x2_out, y2_out = self.apply_zin(x1, y1, x2, y2, curr_step)
        elif mvtint == -3:
            x1_out, y1_out, x2_out, y2_out = self.apply_zout(x1, y1, x2, y2, curr_step)
        elif mvtint == 4:
            x1_out, y1_out, x2_out, y2_out = self.apply_left(x1, y1, x2, y2, curr_step)
            x1_out, y1_out, x2_out, y2_out = self.apply_zin(x1_out, y1_out, x2_out, y2_out, curr_step)
        elif mvtint == -4:
            x1_out, y1_out, x2_out, y2_out = self.apply_right(x1, y1, x2, y2, curr_step)
            x1_out, y1_out, x2_out, y2_out = self.apply_zout(x1_out, y1_out, x2_out, y2_out, curr_step)
        elif mvtint == 5:
            x1_out, y1_out, x2_out, y2_out = self.apply_right(x1, y1, x2, y2, curr_step)
            x1_out, y1_out, x2_out, y2_out = self.apply_zin(x1_out, y1_out, x2_out, y2_out, curr_step)
        elif mvtint == -5:
            x1_out, y1_out, x2_out, y2_out = self.apply_left(x1, y1, x2, y2, curr_step)
            x1_out, y1_out, x2_out, y2_out = self.apply_zout(x1_out, y1_out, x2_out, y2_out, curr_step)
        elif mvtint == 6:
            x1_out, y1_out, x2_out, y2_out = self.apply_up(x1, y1, x2, y2, curr_step)
            x1_out, y1_out, x2_out, y2_out = self.apply_zin(x1_out, y1_out, x2_out, y2_out, curr_step)
        elif mvtint == -6:
            x1_out, y1_out, x2_out, y2_out = self.apply_down(x1, y1, x2, y2, curr_step)
            x1_out, y1_out, x2_out, y2_out = self.apply_zout(x1_out, y1_out, x2_out, y2_out, curr_step)
        elif mvtint == 7:
            x1_out, y1_out, x2_out, y2_out = self.apply_down(x1, y1, x2, y2, curr_step)
            x1_out, y1_out, x2_out, y2_out = self.apply_zin(x1_out, y1_out, x2_out, y2_out, curr_step)
        elif mvtint == -7:
            x1_out, y1_out, x2_out, y2_out = self.apply_up(x1, y1, x2, y2, curr_step)
            x1_out, y1_out, x2_out, y2_out = self.apply_zout(x1_out, y1_out, x2_out, y2_out, curr_step)
        return x1_out, y1_out, x2_out, y2_out

    def apply_left(self, x1, y1, x2, y2, curr_step):
        w = x2 - x1
        dw = int(w / self.step * curr_step) 
        x1_out = min(x1 + dw, self.fullwidth)
        x2_out = min(x2 + dw, self.fullwidth)
        y1_out = y1
        y2_out = y2
        return x1_out, y1_out, x2_out, y2_out

    def apply_right(self, x1, y1, x2, y2, curr_step):
        w = x2 - x1
        dw = int(w / self.step * curr_step) 
        x1_out = max(x1 - dw, 0)
        x2_out = max(x2 - dw, 0)
        y1_out = y1
        y2_out = y2
        return x1_out, y1_out, x2_out, y2_out

    def apply_down(self, x1, y1, x2, y2, curr_step):
        h = y2 - y1
        dh = int(h / self.step * curr_step) 
        x1_out = x1
        x2_out = x2
        y1_out = min(y1 + dh, self.fullheight)
        y2_out = min(y2 + dh, self.fullheight)
        return x1_out, y1_out, x2_out, y2_out

    def apply_up(self, x1, y1, x2, y2, curr_step):
        h = y2 - y1
        dh = int(h / self.step * curr_step) 
        x1_out = x1
        x2_out = x2
        y1_out = max(y1 - dh, 0)
        y2_out = max(y2 - dh, 0)
        return x1_out, y1_out, x2_out, y2_out

    def apply_zin(self, x1, y1, x2, y2, curr_step):
        w = x2 - x1
        h = y2 - y1
        dw = int((w / self.step * curr_step)//2)
        dh = int((h / self.step * curr_step)//2)

        x1_out = max(x1 - w//3 + dw, 0)
        x2_out = min(x2 + w//3 - dw, self.fullwidth)
        y1_out = max(y1 - h//3 + dh, 0)
        y2_out = min(y2 + h//3 - dh, self.fullheight)
        return x1_out, y1_out, x2_out, y2_out

    def apply_zout(self, x1, y1, x2, y2, curr_step):
        w = x2 - x1
        h = y2 - y1
        dw = int((w / self.step * curr_step)//2)
        dh = int((h / self.step * curr_step)//2)

        x1_out = max(x1 - dw, 0)
        x2_out = min(x2 + dw, self.fullwidth)
        y1_out = max(y1 - dh, 0)
        y2_out = min(y2 + dh, self.fullheight)
        return x1_out, y1_out, x2_out, y2_out

    def post_crop_processing(self, x1, y1, x2, y2):
        # padding
        y1 = max((y1+y2)*0.5 - (y2-y1)*0.5*self.padding_scale,0)
        y2 = min((y1+y2)*0.5 + (y2-y1)*0.5*self.padding_scale,height)
        x1 = max((x1+x2)*0.5 - (x2-x1)*0.5*self.padding_scale,0)
        x2 = min((x1+x2)*0.5 + (x2-x1)*0.5*self.padding_scale,width)

        # recrop to fit aspect ratio
        if (x2-x1)/(y2-y1) > self.fullwidth/self.fullheight:
            ratio = ((x2-x1)/(y2-y1))/(self.fullwidth/self.fullheight)
            y1_ = max((y1+y2)*0.5 - (y2-y1)*0.5*ratio,0)
            y2_ = min((y1+y2)*0.5 + (y2-y1)*0.5*ratio,self.fullheight)
            if y1_ == 0:
                y2_ = (self.fullheight/self.fullwidth)*(x2-x1)
            elif y2_ == self.fullheight:
                y1_ = self.fullheight - (self.fullheight/self.fullwidth)*(x2-x1)
            y1, y2 = y1_, y2_
        elif (x2-x1)/(y2-y1) < self.fullwidth/self.fullheight:
            ratio = ((y2-y1)/(x2-x1))*(self.fullwidth/self.fullheight)
            ratio
            x1_ = max((x1+x2)*0.5 - (x2-x1)*0.5*ratio,0)
            x2_ = min((x1+x2)*0.5 + (x2-x1)*0.5*ratio,self.fullwidth)
            if x1_ == 0:
                x2_ = (self.fullwidth/self.fullheight)*(y2-y1)
            elif x2_ == self.fullwidth:
                x1_ = self.fullwidth - (self.fullwidth/self.fullheight)*(y2-y1)
            x1, x2 = x1_, x2_
        return x1, y1, x2, y2

if __name__ == '__main__':
    ## 设置参数
    # 视频存放、导出路径
    data_root = ''
    # 场景理解pkl
    pkl1 = '飞书20220129-210653_labeled_smooth.pkl'
    pkl2 = '飞书20220129-210549_labeled_smooth.pkl'
    # 原始视频名称，注：下面list中第一个一定要放大全机位
    #src_list = ['A30_rec.mp4','L30.mp4','R30.mp4']
    src_list = ['source_media/A2.mp4','source_media/L2.mp4','source_media/R2.mp4']
    output_name = 'out.mp4'
    output_path = os.path.join(data_root, output_name)
    w = 1920
    h = 1080
    # 是否导入音轨
    audio_flag = False
    # 音频路径
    bgm_path = 'source_media/bgm.mp3'
    # 是否增加单人特写剪裁
    crop_flag = True
    # 是否添加运镜：只有在裁剪模式下才可以运镜，运镜分zoom in / zoom out / shift left / shift right / shift up /shift down，运镜方式可组合
    len_mv_flag = True
    # 可能的裁剪情况：大全（开场谢幕用）、每个人员的特写、舞台所有人全画幅展示（左右机位）、多个演员之间的互动（后面再说）
    actor_amount =  3
    position_amount = 3
    assert position_amount==len(src_list), "错误提示：机位个数和输入视频数不等"
    # 开头和结尾slice帧数，初始化75，大约各3s大全，可改
    res_frame = 75
    # 每个slice平均时长，假定每个slice的平均时长是6s（约150帧），可改
    avg_stay = 150
    # 每个slice的最小时长，e.g., 4s，可改
    min_stay = 100
    # 演员id初始化 Tips: 更新底库需重写
    actor_id_list = [0,1,2]
    # 检测框向外扩展系数，可改
    padding_scale = 1.1
    # lens movement total steps, for slower movement put larger steps
    len_mv_steps = 300

    ## 随机初始化
    reader = imageio.get_reader(os.path.join(data_root, src_list[0]))
    fps = reader.get_meta_data()['fps']
    cap=cv2.VideoCapture(os.path.join(data_root, src_list[0]))
    total_frame_num=int(cap.get(7))
    print(total_frame_num)
    # 时间分段，初始化一个列表来记录每一小段id
    slice_id = np.zeros(total_frame_num)
    slice_id[:res_frame] = 0
    slice_id[-res_frame:] = 0
    # 根据平均时长，确定中间一共切镜多少次
    change_times = int((total_frame_num - res_frame*2)/avg_stay)
    # 每次改变机位最少停留4s(100帧）才可能再次改变，其他随机
    for i in range(change_times):
        slice_id[res_frame+min_stay*i:min_stay+res_frame+min_stay*i] = i+1
    for i in range((res_frame+change_times*min_stay),(total_frame_num-res_frame)):
        slice_id[i] = np.random.randint(1,change_times+1)
    slice_id[res_frame:-res_frame] = np.sort(slice_id[res_frame:-res_frame])
    # 机位id初始化
    position_id = slice_id.copy()
    for i in range(1,int(np.max(slice_id))+1):
        position_id[slice_id==i] = np.random.randint(1,len(src_list))
    position_id = position_id.astype(np.int32)
    print(np.unique(position_id))
    # 裁剪方式初始化
    # 可能的裁剪情况：每个人员的特写（演员id，0-2)、大全（开场谢幕用）4、舞台所有人全画幅展示（左右机位）3
    crop_id = slice_id.copy()
    crop_id[slice_id==0] = len(actor_id_list)+2
    
    for i in range(1,int(np.max(slice_id))+1):
        crop_id[slice_id==i] = np.random.randint(0,len(actor_id_list)+1)
    
    print(np.unique(crop_id))

    # 运镜方式初始化，可组合：0 None, 1 left, -1 right, 2 up, -2 down, 3 zoomin, -3 zoomout, 4 left zoomin, -4 right zoomout, 5 right zoomin, -5 left zoomout, 6 up zoomin, -6 down zoomout, 7 down zoomin, -7 up zoomout 
    len_mv_id = crop_id.copy()
    len_mv_id[:res_frame] = 7 # 开幕 down zoom in
    len_mv_id[-res_frame:] = -6 # 谢幕 down zoom out
    
    for i in range(1,int(np.max(slice_id))+1):
        len_mv_id[slice_id==i] = np.random.randint(-7, 7)
    print(np.unique(len_mv_id))

    # 读取场景理解信息
    df1=pd.read_pickle(pkl1)
    df2=pd.read_pickle(pkl2)
    df_list = [df1,df2]

    ## 写MP4
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))
    flag_start = True
    curr_step = 0 # for lens movement
    for frame_cur_num in tqdm(range(total_frame_num)):
        frame_cur_num = int(frame_cur_num)
        # 确定当前时刻是否换机位，如果不换则延续上一帧reader | # update curr_step for lens movement if crop_id changes
        if flag_start:
            flag_start = False
            # print(position_id[frame_cur_num])
            cap = cv2.VideoCapture(os.path.join(data_root, src_list[position_id[frame_cur_num]])) 
            _,img=cap.read()
            height, width, _ = img.shape
        elif position_id[frame_cur_num] != position_id[frame_cur_num-1]:
            cap = cv2.VideoCapture(os.path.join(data_root, src_list[position_id[frame_cur_num]]))
            curr_step = 0 
        elif crop_id[frame_cur_num] != crop_id[frame_cur_num-1]:
            curr_step = 0
        else:
            curr_step = curr_step + 1

        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_cur_num)  #设置要获取的帧号
        _,img=cap.read()  #read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        len_mv = MvLens(width, height, step=len_mv_steps, padding_scale=padding_scale) # larger step, slower 运镜

        # 对img进行运镜、裁剪、resize 
        # 如果需要单人特写：裁剪 + 画质增强
        if crop_flag and crop_id[frame_cur_num]<=2:
            # get bounding boxes coords
            df = df_list[position_id[frame_cur_num]-1]
            info = df[(df['actor_id']==crop_id[frame_cur_num])&(df['frameid']==str(frame_cur_num+1))]
            info = info.reset_index(drop=True)
            x1,y1,x2,y2 = float(info['x1'][0])*width,float(info['y1'][0])*height,float(info['x2'][0])*width,float(info['y2'][0])*height
        else: # deal with 全景 
            x1 = width//4
            y1 = height//4
            x2 = width//4 * 3
            y2 = height//4 * 3

        # apply lens movement
        x1_, y1_, x2_, y2_ = len_mv.apply_lensmv(x1, y1, x2, y2, curr_step, len_mv_id[frame_cur_num])
        
        # post crop processing: padding + ensure aspect ratio
        x1_, y1_, x2_, y2_ = len_mv.post_crop_processing(x1_, y1_, x2_, y2_) # after len mvt

        # check if body bbox still present after len mv
        if crop_flag and crop_id[frame_cur_num]<=2:
            if (x1_ > x1) or (y1_ > y1) or (x2_ < x2) or (y2_ < y2):
                curr_step = curr_step - 1 # stop lens movement

        x1, y1, x2, y2 = x1_, y1_, x2_, y2_

        img = img[int(y1):int(y2),int(x1):int(x2),:]

        out = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        video_writer.write(out)

    ## 添加音轨
    if audio_flag:
        audio = AudioFileClip(bgm_path)
        output_path_withaudio = os.path.join(data_root, output_name[:-4]+'_audio.mp4')
        video = VideoFileClip(output_path)# 读入视频
        video = video.set_audio(audio)# 将音轨合成到视频中
        video.write_videofile(output_path_withaudio,audio_codec="aac")# 输出

