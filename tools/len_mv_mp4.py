import cv2, os
from moviepy import *
from moviepy.editor import *
import numpy as np
import imageio
import pandas as pd
class VideoGenerator:
    def __init__(self, width=1920, height=1080, step=300):
        self.width = width
        self.height = height
        self.step = step


    def linear(self, image_path, video_path):
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 25.6, (self.width, self.height))
        image = cv2.imread(image_path)
        h, w, c = image.shape
        for i in range(self.step):
            dh = (h - self.height) // 2
            dw = int((w - self.width) / self.step * i)
            out = image[dh:dh + self.height, dw:dw + 1920, :]
            video_writer.write(out)

    def center_zoomout(self, image_path, video_path):
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 25.6, (self.width, self.height))
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        for i in range(self.step):
            ch = self.height + int((h - self.height) / self.step * i)
            cw = self.width + int((w - self.width) / self.step * i)
            dh = (h - ch) // 2
            dw = (w - cw) // 2
            cim = image[dh:dh + ch, dw:dw + cw, :]
            out = cv2.resize(cim, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)

            video_writer.write(out)

    def left_zoomout(self, image_path, video_path):
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 25.6, (self.width, self.height))
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        scale_h = h / self.height
        scale_w = w / self.width
        largest_scale = min(scale_h, scale_w)
        scale_step = (largest_scale - 1) / self.step
        largest_w = w - self.width * largest_scale
        w_step = largest_w / self.step
        for i in range(self.step):
            cscale = scale_step * i + 1.0
            ch = int(self.height * cscale)
            cw = int(self.width * cscale)
            dh = (h - ch) // 2
            dw = int(i * w_step)
            cim = image[dh:dh + ch, dw:dw + cw, :]
            out = cv2.resize(cim, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)

            video_writer.write(out)

    def move_right_zoomin(self, image_path, video_path):
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 25.6, (self.width, self.height))
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        tar_scale = 0.8
        scale_step = (tar_scale - 1.0) / self.step
        tar_width = self.width * tar_scale
        #tar_height = 1080 * tar_scale
        w_step = (w - tar_width) / self.step

        for i in range(self.step):
            ch = int(self.height * (1.0 + scale_step * i))
            cw = int(self.width * (1.0 + scale_step * i))
            dh = (h - ch) // 2
            dw = int(i * w_step)
            cim = image[dh:dh + ch, dw:dw + cw, :]
            out = cv2.resize(cim, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
            #print(ch, cw, dh, dw, dw + cw)

            video_writer.write(out)
        print(video_path)

if __name__ == '__main__':
    gen = VideoGenerator(step=256)
    ## 设置参数
    # 视频存放、导出路径
    data_root = '/mnt/bd/lwlq-workshop/yolov5/信天游/'
    # 场景理解pkl
    pkl1 = '/mnt/bd/lwlq-workshop/Person_reID_baseline_pytorch/runs/detect/exp3/labels/飞书20220129-210653_labeled_smooth.pkl'
    pkl2 = '/mnt/bd/lwlq-workshop/Person_reID_baseline_pytorch/runs/detect/exp/labels/飞书20220129-210549_labeled_smooth.pkl'
    # 原始视频名称，注：下面list中第一个一定要放大全机位
    src_list = ['飞书20220129-210659.mp4','飞书20220129-210653.mp4','飞书20220129-210549.mp4']
    output_name = 'out.mp4'
    output_path = os.path.join(data_root, output_name)
    w = 1920
    h = 1080
    # 是否导入音轨
    audio_flag = False
    # 音频路径
    bgm_path = '/mnt/bd/lwlq-workshop/yolov5/信天游/bgm.mp3'
    # 是否增加单人特写剪裁
    crop_flag = True
    # 是否添加运镜：只有在裁剪模式下才可以运镜，运镜分zoom in / zoom out / shift left / shift right / shift up /shift down，运镜方式可组合
    len_mv_flag = True
    # 可能的裁剪情况：大全（开场谢幕用）、每个人员的特写、舞台所有人全画幅展示（左右机位）、多个演员之间的互动
    actor_amount =  3
    position_amount = 3
    assert position_amount==len(src_list), "错误提示：机位个数和输入视频数不等"
    # 开头和结尾slice帧数，初始化75，大约各3s大全，可改
    res_frame = 75
    # 每个slice平均时长，假定每个slice的平均时长是6s（约150帧）
    avg_stay = 150
    # 每个slice的最小时长，e.g., 4s
    min_stay = 100
    # 演员id初始化 Tips: 更新底库需重写
    actor_id_list = [0,1,2]
    # 检测框向外扩展系数
    padding_scale = 1.1

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
        position_id[position_id==i] = np.random.randint(1,len(src_list))
    position_id = position_id.astype(np.int32)
    print(np.unique(position_id))
    # 裁剪方式初始化
    # 可能的裁剪情况：每个人员的特写（演员id，0-2)、大全（开场谢幕用）4、舞台所有人全画幅展示（左右机位）3
    crop_id = position_id.copy()
    crop_id[crop_id==0] = len(actor_id_list)+2
    for i in range(1,int(np.max(slice_id))+1):
        crop_id[crop_id==i] = np.random.randint(0,len(actor_id_list)+1)
    print(np.unique(crop_id))

    # 运镜方式初始化，可组合：zoom in / zoom out / shift left / shift right / shift up / shift down / 0 None
    len_mv_id = crop_id.copy()
    len_mv_id[len_mv_id == (len(actor_id_list)+2)] = 0
    

    # 读取场景理解信息
    df1=pd.read_pickle(pkl1)
    df2=pd.read_pickle(pkl2)
    df_list = [df1,df2]

    ## 写MP4
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))
    flag_start = True
    for frame_cur_num in range(total_frame_num):
        frame_cur_num = int(frame_cur_num)
        # 确定当前时刻是否换机位，如果不换则延续上一帧reader
        if flag_start:
            flag_start = False
            # print(position_id[frame_cur_num])
            cap = cv2.VideoCapture(os.path.join(data_root, src_list[position_id[frame_cur_num]])) 
            _,img=cap.read()
            height, width, _ = img.shape
        elif position_id[frame_cur_num] != position_id[frame_cur_num-1]:
            cap = cv2.VideoCapture(os.path.join(data_root, src_list[position_id[frame_cur_num]])) 


        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_cur_num)  #设置要获取的帧号
        _,img=cap.read()  #read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True

        # 对img进行运镜、裁剪、resize 
        # 如果需要单人特写：裁剪 + 去抖 + SR 
        if crop_flag and crop_id[frame_cur_num]<=2:
            df = df_list[position_id[frame_cur_num]-1]
            info = df[(df['actor_id']==crop_id[frame_cur_num])&(df['frameid']==str(frame_cur_num+1))]
            info = info.reset_index(drop=True)
            x1,y1,x2,y2 = float(info['x1'][0])*width,float(info['y1'][0])*height,float(info['x2'][0])*width,float(info['y2'][0])*height
            y1 = max((y1+y2)*0.5 - (y2-y1)*0.5*padding_scale,0)
            y2 = min((y1+y2)*0.5 + (y2-y1)*0.5*padding_scale,height)
            x1 = max((x1+x2)*0.5 - (x2-x1)*0.5*padding_scale,0)
            x2 = min((x1+x2)*0.5 + (x2-x1)*0.5*padding_scale,width)
            if (x2-x1)/(y2-y1) > width/height:
                ratio = ((x2-x1)/(y2-y1))/(width/height)
                y1_ = max((y1+y2)*0.5 - (y2-y1)*0.5*ratio,0)
                y2_ = min((y1+y2)*0.5 + (y2-y1)*0.5*ratio,height)
                if y1_ == 0:
                    y2_ = (height/width)*(x2-x1)
                elif y2_ == height:
                    y1_ = height - (height/width)*(x2-x1)
                y1, y2 = y1_, y2_
            elif (x2-x1)/(y2-y1) < width/height:
                ratio = ((y2-y1)/(x2-x1))*(width/height)
                ratio
                x1_ = max((x1+x2)*0.5 - (x2-x1)*0.5*ratio,0)
                x2_ = min((x1+x2)*0.5 + (x2-x1)*0.5*ratio,width)
                if x1_ == 0:
                    x2_ = (width/height)*(y2-y1)
                elif x2_ == width:
                    x1_ = width - (width/height)*(y2-y1)
                x1, x2 = x1_, x2_
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

