"""
python tools/len_mv_mp4.py --srcvidfiles align-videos/A_5min_large2.mp4 align-videos/L_5min_large2.mp4 align-videos/R_5min_large2.mp4 --output_name out_5min_ts_conf075_offl15r91_lens400.mp4 --pklfiles L_5min_labeled_smooth.pkl R_5min_labeled_smooth.pkl --crop_flag --len_mv_flag --len_mv_steps 400 --bbox_conf_thresh 0.75 --q_sz 256 --decode_sleep 1.0 --audioseg_pkl audio_segmentation/5min_audioseg.pkl
"""
import cv2, os
import numpy as np
import imageio
from tqdm import tqdm
import pandas as pd
import argparse
import os
from threading import Thread
from queue import Queue
import time
from videoreader import VideoReader

# main video processing
def process_video():
    ## 写MP4
    video_writer = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))
    flag_start = True
    # process frames in current group
    for frame_cur_num in tqdm(range(total_frame_num)):
        # init and update parameters at start of each frame
        frame_cur_num = int(frame_cur_num)
        curr_slice_id = slice_id[frame_cur_num]

        if flag_start: # start of every group of frames, might not be the beginning of a slice
            flag_start = False
            curr_step = frame_cur_num - list(slice_id).index(curr_slice_id) # init curr_step for lens movement, curr_step=0 at the start of each slice 
            if position_id[frame_cur_num] == 0:
                fvs = FileVideoStream(os.path.join(data_root, src_list[position_id[frame_cur_num]]), queueSize=q_sz, startframenum=frame_cur_num, num_frames_tot=dict_numframesperslice[curr_slice_id]).start()
            elif position_id[frame_cur_num] == 1:
                fvs = FileVideoStream(os.path.join(data_root, src_list[position_id[frame_cur_num]]), queueSize=q_sz, startframenum=frame_cur_num, num_frames_tot=dict_numframesperslice[curr_slice_id]).start()
            elif position_id[frame_cur_num] == 2:
                fvs = FileVideoStream(os.path.join(data_root, src_list[position_id[frame_cur_num]]), queueSize=q_sz, startframenum=frame_cur_num, num_frames_tot=dict_numframesperslice[curr_slice_id]).start()
            time.sleep(decode_sleep) # let thread handle video decoding of a batch of frames at one shot
            height,width = fvs.get_heightwidth()
            x1_prev, y1_prev, x2_prev, y2_prev = 0, 0, width, height # init previous coords values
        # 确定当前时刻是否换机位，如果不换则延续上一帧reader 
        elif (slice_id[frame_cur_num] != slice_id[frame_cur_num-1]):
            #cap = cv2.VideoCapture(os.path.join(data_root, src_list[position_id[frame_cur_num]]))
            if position_id[frame_cur_num] == 0:
                fvs = FileVideoStream(os.path.join(data_root, src_list[position_id[frame_cur_num]]), queueSize=q_sz, startframenum=frame_cur_num, num_frames_tot=dict_numframesperslice[curr_slice_id]).start()
            elif position_id[frame_cur_num] == 1:
                fvs = FileVideoStream(os.path.join(data_root, src_list[position_id[frame_cur_num]]), queueSize=q_sz, startframenum=frame_cur_num, num_frames_tot=dict_numframesperslice[curr_slice_id]).start()
            elif position_id[frame_cur_num] == 2:
                fvs = FileVideoStream(os.path.join(data_root, src_list[position_id[frame_cur_num]]), queueSize=q_sz, startframenum=frame_cur_num, num_frames_tot=dict_numframesperslice[curr_slice_id]).start()
            time.sleep(decode_sleep) # let thread handle video decoding
            curr_step = 0 # reset to 0 at start of each slice
        else:
            curr_step = curr_step + 1 # increment for increased lens mvt

        img=fvs.read()  #read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        height, width, _ = img.shape
        len_mv = MvLens(width, height, w, h, padding_scale=padding_scale) # init mvlens

        # 对img进行运镜、裁剪、resize 
        # 如果需要单人特写：裁剪 + 画质增强
        if crop_flag and crop_id[frame_cur_num]<=2: # for 单人特写
            # get bounding boxes coords
            df = df_list[position_id[frame_cur_num]-1]
            info = df[(df['actor_id']==crop_id[frame_cur_num])&(df['frameid']==str(frame_cur_num+1))]
            info = info.reset_index(drop=True)

            found = False # check if intended actor bbox found in current frame
            if info.shape[0] > 0:
                for i in range(info.shape[0]):
                    if float(info['conf'][i]) > bbox_conf_thresh: # only select bbox with high enough confidence to avoid false detections
                        x1,y1,x2,y2 = float(info['x1'][i])*width,float(info['y1'][i])*height,float(info['x2'][i])*width,float(info['y2'][i])*height
                        # prevent large changes in crop box over consecutive frames
                        if [x1_prev, y1_prev, x2_prev, y2_prev] != [0, 0, width, height] and curr_step!=0: 
                            change_thresh = [(x2_prev-x1_prev)*0.1, (y2_prev-y1_prev)*0.1] # set threshold at 10% of bbox width, height
                            if abs(x1-x1_prev)>change_thresh[0] or abs(x2-x2_prev)>change_thresh[0] or abs(y1-y1_prev)>change_thresh[1] or abs(y2-y2_prev)>change_thresh[1]:
                                x1, y1, x2, y2 = x1_prev, y1_prev, x2_prev, y2_prev
                        # update prev coords values
                        x1_prev, y1_prev, x2_prev, y2_prev = x1, y1, x2, y2
                        found = True
                        break           
            if not found: # if intended actor bbox not found in this frame
                if curr_step != 0:
                    x1, y1, x2, y2 = x1_prev, y1_prev, x2_prev, y2_prev # use bbox from previous frame
                else: # if it's the start of the slice, then swap the intended actor for 左右全景
                    if position_id[frame_cur_num] == 1: #L
                        x1,y1,x2,y2 = wide_crop_boxes[1]
                        x1,y1,x2,y2 = x1*width, y1*height, x2*width, y2*height
                    elif position_id[frame_cur_num] == 2: #R
                        x1,y1,x2,y2 = wide_crop_boxes[2]
                        x1,y1,x2,y2 = x1*width, y1*height, x2*width, y2*height
                    crop_id[slice_id==curr_slice_id] = 3 # set to 左右全景
                    len_mv_id[slice_id==curr_slice_id] = 0 # set to still
        elif crop_id[frame_cur_num] ==3:
            if position_id[frame_cur_num] == 1: #L
                x1,y1,x2,y2 = wide_crop_boxes[1]
                x1,y1,x2,y2 = x1*width, y1*height, x2*width, y2*height
            elif position_id[frame_cur_num] == 2: #R
                x1,y1,x2,y2 = wide_crop_boxes[2]
                x1,y1,x2,y2 = x1*width, y1*height, x2*width, y2*height
        else: # for 全景 
            x1,y1,x2,y2 = wide_crop_boxes[0]
            x1,y1,x2,y2 = x1*width, y1*height, x2*width, y2*height
                
        if len_mv_flag:
            # apply lens movement: 目前全部的运镜设置成同样的 speed, but possible to have different speed for each slice
            if halfbody_id[frame_cur_num] and crop_flag and crop_id[frame_cur_num]<=2: 
                x1_, y1_, x2_, y2_ = len_mv.apply_lensmv(x1, y1, x2, y2 - (y2-y1)/2, curr_step, len_mv_id[frame_cur_num], len_mv_steps)
            elif h > w and mode_wideshot==3:
                x1_, y1_, x2_, y2_ = len_mv.apply_lensmv(x1, y1, x2, y2, curr_step, 1, 60) # right panning
            else:
                x1_, y1_, x2_, y2_ = len_mv.apply_lensmv(x1, y1, x2, y2, curr_step, len_mv_id[frame_cur_num], len_mv_steps)
            
        # post crop processing: padding + ensure aspect ratio
        x1_p, y1_p, x2_p, y2_p = len_mv.post_crop_processing(x1_, y1_, x2_, y2_) # after len mvt
    
        # check if bbox still present after len mv
        if crop_flag and crop_id[frame_cur_num]<=2:
            if halfbody_id[frame_cur_num]:
                if (x1_p > x1) or (y1_p > y1) or (x2_p < x2) or (y2_p < y2/2):
                    curr_step = curr_step - 1 # stop lens movement
            else:
                if (x1_p > x1) or (y1_p > y1) or (x2_p < x2) or (y2_p < y2):
                    curr_step = curr_step - 1 # stop lens movement
        elif crop_id[frame_cur_num] == 3 and mode_wideshot!=3:
            if (x1_p > x1)  or (x2_p < x2):
                curr_step = curr_step - 1 # stop lens movement
        
        # final output frame to be rendered
        if h > w and crop_id[frame_cur_num] > 2 and mode_wideshot!=3: # vertical mode dealing with wide shots
            out = len_mv.vertmode_wideshot_processing(x1_, y1_, x2_, y2_, mode_wideshot, img) # coordinates before postprocessing
        else:
            img = img[int(y1_p):int(y2_p),int(x1_p):int(x2_p),:] # int might introduce rounding errors?
            out = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        video_writer.write(out)


class MvLens: # for lens movement, and post-processing (padding + ensuring aspect ratio)
    def __init__(self, fullwidth, fullheight, des_width, des_height, padding_scale=1.1):
        self.fullwidth = fullwidth # width of source video image
        self.fullheight = fullheight # height of source video image
        self.des_width = des_width # desired width of output video
        self.des_height = des_height # desired height of output video
        self.padding_scale = padding_scale

    def apply_lensmv(self, x1, y1, x2, y2, curr_step, mvtint, total_steps):
        # x1,y1,x2,y2 are the coords of the rectangular crop intended to be at the centre of the output frame
        # mvtint: 0 None, 1 left, -1 right, 3 zoomin, -3 zoomout, 4 left zoomin, -4 right zoomout, 5 right zoomin, -5 left zoomout
        # total_steps affects speed of 运镜, greater total_steps means slower 运镜
        if mvtint == 0:
            x1_out, y1_out, x2_out, y2_out = x1, y1, x2, y2
        elif mvtint == 1:
            x1_out, y1_out, x2_out, y2_out = self.apply_left(x1, y1, x2, y2, curr_step, total_steps)
        elif mvtint == -1:
            x1_out, y1_out, x2_out, y2_out = self.apply_right(x1, y1, x2, y2, curr_step, total_steps)
        elif mvtint == 3:
            x1_out, y1_out, x2_out, y2_out = self.apply_zin(x1, y1, x2, y2, curr_step, total_steps)
        elif mvtint == -3:
            x1_out, y1_out, x2_out, y2_out = self.apply_zout(x1, y1, x2, y2, curr_step, total_steps)
        elif mvtint == 4:
            x1_out, y1_out, x2_out, y2_out = self.apply_left(x1, y1, x2, y2, curr_step, total_steps)
            x1_out, y1_out, x2_out, y2_out = self.apply_zin(x1_out, y1_out, x2_out, y2_out, curr_step, total_steps)
        elif mvtint == -4:
            x1_out, y1_out, x2_out, y2_out = self.apply_right(x1, y1, x2, y2, curr_step, total_steps)
            x1_out, y1_out, x2_out, y2_out = self.apply_zout(x1_out, y1_out, x2_out, y2_out, curr_step, total_steps)
        elif mvtint == 5:
            x1_out, y1_out, x2_out, y2_out = self.apply_right(x1, y1, x2, y2, curr_step, total_steps)
            x1_out, y1_out, x2_out, y2_out = self.apply_zin(x1_out, y1_out, x2_out, y2_out, curr_step, total_steps)
        elif mvtint == -5:
            x1_out, y1_out, x2_out, y2_out = self.apply_left(x1, y1, x2, y2, curr_step, total_steps)
            x1_out, y1_out, x2_out, y2_out = self.apply_zout(x1_out, y1_out, x2_out, y2_out, curr_step, total_steps)
        return x1_out, y1_out, x2_out, y2_out

    def apply_left(self, x1, y1, x2, y2, curr_step, total_steps):
        w = x2 - x1
        dw = int(w / total_steps * curr_step) 
        x1_out = min(x1 + dw, self.fullwidth)
        x2_out = min(x2 + dw, self.fullwidth)
        y1_out = y1
        y2_out = y2
        return x1_out, y1_out, x2_out, y2_out

    def apply_right(self, x1, y1, x2, y2, curr_step, total_steps):
        w = x2 - x1
        dw = int(w / total_steps * curr_step) 
        x1_out = max(x1 - dw, 0)
        x2_out = max(x2 - dw, 0)
        y1_out = y1
        y2_out = y2
        return x1_out, y1_out, x2_out, y2_out

    def apply_zin(self, x1, y1, x2, y2, curr_step, total_steps):
        w = x2 - x1
        h = y2 - y1
        dw = int((w / total_steps * curr_step)//2)
        dh = int((h / total_steps * curr_step)//2)

        x1_out = min(max(x1 - w//3 + dw, 0), self.fullwidth)
        x2_out = max(min(x2 + w//3 - dw, self.fullwidth), 0)
        y1_out = min(max(y1 - h//3 + dh, 0), self.fullheight)
        y2_out = max(min(y2 + h//3 - dh, self.fullheight), 0)
        return x1_out, y1_out, x2_out, y2_out

    def apply_zout(self, x1, y1, x2, y2, curr_step, total_steps):
        w = x2 - x1
        h = y2 - y1
        dw = int((w / total_steps * curr_step)//2)
        dh = int((h / total_steps * curr_step)//2)

        x1_out = max(x1 - dw, 0)
        x2_out = min(x2 + dw, self.fullwidth)
        y1_out = max(y1 - dh, 0)
        y2_out = min(y2 + dh, self.fullheight)
        return x1_out, y1_out, x2_out, y2_out

    def post_crop_processing(self, x1, y1, x2, y2):
        # padding
        y1 = max((y1+y2)*0.5 - (y2-y1)*0.5*self.padding_scale,0)
        y2 = min((y1+y2)*0.5 + (y2-y1)*0.5*self.padding_scale,self.fullheight)
        x1 = max((x1+x2)*0.5 - (x2-x1)*0.5*self.padding_scale,0)
        x2 = min((x1+x2)*0.5 + (x2-x1)*0.5*self.padding_scale,self.fullwidth)

        # recrop to fit aspect ratio
        if (x2-x1)/(y2-y1) > self.des_width/self.des_height:
            ratio = ((x2-x1)/(y2-y1))/(self.des_width/self.des_height)
            y1_ = max((y1+y2)*0.5 - (y2-y1)*0.5*ratio,0)
            y2_ = min((y1+y2)*0.5 + (y2-y1)*0.5*ratio,self.fullheight)
            if y1_ == 0:
                y2_ = (self.des_height/self.des_width)*(x2-x1)
            elif y2_ == self.fullheight:
                y1_ = self.fullheight - (self.des_height/self.des_width)*(x2-x1)
            y1, y2 = y1_, y2_
        elif (x2-x1)/(y2-y1) < self.des_width/self.des_height:
            ratio = ((y2-y1)/(x2-x1))*(self.des_width/self.des_height)
            x1_ = max((x1+x2)*0.5 - (x2-x1)*0.5*ratio,0)
            x2_ = min((x1+x2)*0.5 + (x2-x1)*0.5*ratio,self.fullwidth)
            if x1_ == 0:
                x2_ = (self.des_width/self.des_height)*(y2-y1)
            elif x2_ == self.fullwidth:
                x1_ = self.fullwidth - (self.des_width/self.des_height)*(y2-y1)
            x1, x2 = x1_, x2_
        return x1, y1, x2, y2

    def vertmode_wideshot_processing(self, x1, y1, x2, y2, mode_int, img):
        """
        set how 全景 is diplayed in vertical mode: 0 - padding with black pixels; 1 - padding with reflection; 2 - blur background; 3 - slide and move across scene
        """
        des_width, des_height = self.des_width, self.des_height
        tmp_width, tmp_height = des_width, des_height//3 # dimensions after splitting vertical screen into 3
        self.des_width, self.des_height = tmp_width, tmp_height
        x1_,y1_,x2_,y2_ = self.post_crop_processing(x1,y1,x2,y2)
        middle_scene = img[int(y1_):int(y2_),int(x1_):int(x2_),:] 
        middle_scene = cv2.resize(middle_scene, (tmp_width, tmp_height), interpolation=cv2.INTER_LANCZOS4)
        if mode_int == 0: # Pad zeros on top and below
            vert_output = cv2.copyMakeBorder(middle_scene,tmp_height,tmp_height, 0, 0,cv2.BORDER_CONSTANT,value=[0,0,0])
        elif mode_int == 1: # Pad with reflection
            vert_output = cv2.copyMakeBorder(middle_scene,tmp_height,tmp_height, 0, 0,cv2.BORDER_REFLECT)
        elif mode_int == 2: # Blur background
            # take centre des_width x des_height crop from entire image
            self.des_width, self.des_height = des_width, des_height
            x1_,y1_,x2_,y2_ = self.post_crop_processing(x1*1.2,y1*2,x2*0.8,y2*0.8)
            background = img[int(y1_):int(y2_),int(x1_):int(x2_),:] 
            background = cv2.resize(background, (des_width, des_height), interpolation=cv2.INTER_LANCZOS4)

            kernelsz = min(des_width, des_height)//10 * 2 + 1
            vert_output = cv2.GaussianBlur(background, (kernelsz, kernelsz),0)
            vert_output[des_height//2 - tmp_height//2:des_height//2 + tmp_height//2, :, :] = middle_scene
        elif mode_int == 3: # Slide: implemented in process_video()
            pass 
        elif mode_int == 4:
            kernelsz = min(tmp_width, tmp_height)//7 * 2 + 1
            bg = cv2.GaussianBlur(middle_scene, (kernelsz, kernelsz),0)
            vert_output = np.vstack((bg, middle_scene, bg))
        
        return vert_output


class FileVideoStream:
    def __init__(self, path, queueSize=128, startframenum=0, num_frames_tot=0):
        # initialize the file video stream along with the boolean
        # # used to indicate if the thread should be stopped or not
        self.stream = VideoReader(path)
        self.stream.seek(startframenum)
        self.stopped = False
        self.num_frames_curr = 0
        self.num_frames_tot = num_frames_tot
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        self.path = path

    def get_heightwidth(self):
        cap = cv2.VideoCapture(self.path)
        return cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH) 

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                frame = self.stream.read()

                if frame is None:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)
                self.num_frames_curr += 1
                if self.num_frames_curr==self.num_frames_tot:
                    self.stop()
                    return
                

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcvidfiles", required=True,  action='store', type=str, nargs=3, help='大全、左、右机位的 source videos, 第一个一定要是大全')
    ap.add_argument("--output_name", required=True, type=str, default="out.mp4", help='output filename')
    ap.add_argument("--pklfiles", required=True,  action='store', type=str, nargs=2, help='smoothing 之后的 pkl file, 分别是左、右机位的 order')
    ap.add_argument("--audioseg_pkl", type=str, help='optional, pkl file containing pandas dataframe of time segmentations based on audio, if None then use random time segmentations')
    ap.add_argument("--crop_flag", action='store_true', help='allow 人员特写')
    ap.add_argument("--len_mv_flag", action='store_true', help='allow 运镜效果')
    ap.add_argument("--padding_scale", type=float, default=1.1, help='how much to pad relative to height and width')
    ap.add_argument("--len_mv_steps", type=int, default=400, help='number of frames over which 运镜 occurs, for slower movement enter larger value')
    ap.add_argument("--bbox_conf_thresh", type=float, default=0.75, help='bbox detection confidence threshold from detect.py, only crops with confidence higher than threshold will be selected')
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--w", type = int, default=1920)
    ap.add_argument("--q_sz", action='store', type=int, default=256, help='buffer size to store video frames, depends on video res and system memory')
    ap.add_argument("--decode_sleep", action='store', type=float, default=1.0, help='time(s) to sleep for thread to decode the video frames in buffer')
    ap.add_argument("--mode_wideshot", type=int, default=0, help='set how 全景 is diplayed in vertical mode: 0 - padding with black pixels; 1 - padding with reflection; 2 - blur background; 3 - slide and move across scene')
    args = vars(ap.parse_args())

    ## 设置参数
    # 视频存放、导出路径
    data_root = ""
    # 原始视频名称，注：下面list中第一个一定要放大全机位
    src_list = args["srcvidfiles"]
    output_name = args["output_name"] #'out.mp4'
    output_path = os.path.join(data_root, output_name)
    # 场景理解pkl
    pkl1, pkl2 = args["pklfiles"]
    pkl_audioseg = args["audioseg_pkl"] #'audio_segmentation/5min_audioseg.pkl'
    w = args["w"] #1920
    h = args["h"] #1080
    # 是否增加单人特写剪裁
    crop_flag = args["crop_flag"] #True
    bbox_conf_thresh = args["bbox_conf_thresh"] #0.65
    # 是否添加运镜：只有在裁剪模式下才可以运镜，运镜分zoom in / zoom out / shift left / shift right / shift up /shift down，运镜方式可组合
    len_mv_flag = args["len_mv_flag"] #True
    # 检测框向外扩展系数，可改
    padding_scale = args["padding_scale"]
    # lens movement total steps, for slower movement put larger steps
    len_mv_steps = args["len_mv_steps"]
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
    # sleep after decoding frame, 否则解帧会一直抢主进程的CPU到100%，不给其他线程CPU空间进行图像预处理和后处理
    decode_sleep = args["decode_sleep"]
    q_sz = args["q_sz"]
    # crop box (x1,y1,x2,y2) for 全景：大全、左、右机位
    wide_crop_boxes = [[0.333, 0.333, 0.667, 1], [0.1, 0.4, 0.9, 1], [0.25, 0.333, 0.875, 1]]
    # vertical mode
    # set how 全景 is diplayed in vertical mode: 0 - padding with black pixels; 1 - blur background; 2 - split screen into 3 actors; 3- split into gray + col + gray, 4 - slide and move across scene
    mode_wideshot = args["mode_wideshot"] # 0, 1, 2, 3, 4
    if mode_wideshot == 3: # slide and move across scene
        wide_crop_boxes = [[0.333, 0.333, 0.444, 1], [0.05, 0.4, 0.3, 1], [0.2, 0.333, 0.4, 1]]
        
   
    ## 随机初始化
    total_frame_num = 1000000000000000000 # random large number
    for i in range(len(src_list)):
        reader = imageio.get_reader(os.path.join(data_root, src_list[i]))
        fps = reader.get_meta_data()['fps']
        print(src_list[i], "has fps of", fps)
        cap=cv2.VideoCapture(os.path.join(data_root, src_list[i]))
        tmp_frame_num = int(cap.get(7)) 
        print(src_list[i], "has framenum of", tmp_frame_num)
        if tmp_frame_num < total_frame_num:
            total_frame_num=tmp_frame_num
    print("Total frame num:", total_frame_num)

    # 时间分段
    if pkl_audioseg is not None:
        # 时间分段 based on audio segmentation
        df_audioseg = pd.read_pickle(pkl_audioseg)
        change_times = df_audioseg.shape[0]

        slice_id = np.zeros(total_frame_num)
        replace_first_t1 = False
        replace_last_t2 = False
        # opening, closing slices
        opening_endframe = int((float(df_audioseg.iloc[0]['t1'])-0.3)*fps)
        if opening_endframe < res_frame:
            opening_endframe = 1
            replace_first_t1 = True

        closing_startframe = int((float(df_audioseg.iloc[-1]['t2'])-0.3)*fps)
        if closing_startframe > total_frame_num - res_frame:
            closing_startframe = total_frame_num
            replace_first_t1 = True
        slice_id[: opening_endframe] = 0
        slice_id[closing_startframe:] = 0
        start_end_frames = [opening_endframe, total_frame_num - closing_startframe]
        # middle slices
        for i in range(change_times):
            if replace_first_t1 and i==0:
                startframe = opening_endframe
            else:
                startframe = int((float(df_audioseg.iloc[i]['t1'])-0.3)*fps)
            if replace_last_t2 and i==change_times-1:
                endframe = total_frame_num
            else:
                endframe = int((float(df_audioseg.iloc[i]['t2'])-0.3)*fps)

            slice_id[startframe:endframe] = i+1
        print('时间分段 based on audio segmentation file, number of slices: ', change_times+2)
    else:
        # 随机时间分段，初始化一个列表来记录每一小段id
        slice_id = np.zeros(total_frame_num)
        slice_id[:res_frame] = 0
        slice_id[-res_frame:] =  0
        # 根据平均时长，确定中间一共切镜多少次
        change_times = int((total_frame_num - res_frame*2)/avg_stay)
        # 每次改变机位最少停留4s(100帧）才可能再次改变，其他随机
        for i in range(change_times):
            slice_id[res_frame+min_stay*i:min_stay+res_frame+min_stay*i] = i+1
        for i in range((res_frame+change_times*min_stay),(total_frame_num-res_frame)):
            slice_id[i] = np.random.randint(1,change_times+1)
        slice_id[res_frame:-res_frame] = np.sort(slice_id[res_frame:-res_frame])
        start_end_frames = [res_frame, res_frame]
        print('随机时间分段, number of slices: ', change_times+2)
    
    # 机位id初始化
    position_id = slice_id.copy()
    for i in range(1,int(np.max(slice_id))+1): 
        position_id[slice_id==i] = np.random.randint(1,len(src_list))
    position_id = position_id.astype(np.int32)
    print('机位id初始化, unique position_ids present: ', np.unique(position_id))

    # 裁剪方式初始化
    # 可能的裁剪情况：每个人员的特写（演员id，0-2)、舞台所有人全画幅展示（左右机位）3、 大全（开场谢幕用）5
    crop_id = slice_id.copy()
    crop_id[slice_id==0] = len(actor_id_list)+2
    for i in range(1,int(np.max(slice_id))+1):
        crop_id[slice_id==i] = np.random.randint(0,len(actor_id_list)+2)
        while crop_id[slice_id==i][0] == crop_id[slice_id==i-1][0] or (crop_id[slice_id==i][0] in [3,4] and crop_id[slice_id==i-1][0] in [3,4]): # make sure its not two consecutive actors at the same time
            crop_id[slice_id==i] = np.random.randint(0,len(actor_id_list)+2)
    crop_id[crop_id==4] = 3
    
    # halfbody 初始化
    halfbody_id = slice_id.copy()
    for i in range(0,int(np.max(slice_id))+1):
        halfbody_id[slice_id==i] = np.random.randint(0,2) # 0: fullbody, 1: halfbody
    print('裁剪方式初始化, unique crop_ids present: ', np.unique(crop_id))

    # 运镜方式初始化，可组合：0 None, 1 left, -1 right, 3 zoomin, -3 zoomout, 4 left zoomin, -4 right zoomout, 5 right zoomin, -5 left zoomout
    len_mv_opts = [0, 1, -1, 3, -3, 4, -4, 5, -5]
    len_mv_opts_halfbody = [3, 4, 5] # only zoomin for halfbody
    len_mv_id = crop_id.copy()
    len_mv_id[:start_end_frames[0]] = 3 # 开幕 zoom in
    len_mv_id[-start_end_frames[1]:] = -3 # 谢幕 zoom out
    if len_mv_flag:
        for i in range(1,int(np.max(slice_id))+1):
            if halfbody_id[slice_id==i][0]:
                len_mv_id[slice_id==i] = len_mv_opts_halfbody[np.random.randint(0, len(len_mv_opts_halfbody))]
            else:
                len_mv_id[slice_id==i] = len_mv_opts[np.random.randint(0, len(len_mv_opts))]
            if crop_id[slice_id==i][0] == 3: # for 左右机位, only allow left right panning
                len_mv_id[slice_id==i] = [-1,1][np.random.randint(0, 2)]
    else:
        len_mv_id = [0]*len(len_mv_id) # no 运镜
    print('运镜方式初始化, unique len_mv_ids present: ', np.unique(len_mv_id))

    i = len(slice_id)-1
    while slice_id[i] == 0:
        slice_id[i] = -1
        i -= 1
    unique, counts = np.unique(slice_id, return_counts=True)
    dict_numframesperslice = dict(zip(unique, counts))
    print(dict_numframesperslice)

    ## 读取场景理解信息
    df1=pd.read_pickle(pkl1)
    df2=pd.read_pickle(pkl2)
    df_list = [df1,df2]

    ## start processing of video frames
    process_video()