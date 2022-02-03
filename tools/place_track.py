import cv2, os
from moviepy import *
from moviepy.editor import *
import numpy as np
import imageio

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
    data_root = '/mnt/bd/lwlq-workshop/yolov5/信天游/'
    # 下面list中第一个一定要放大全机位
    src_list = ['飞书20220129-210659.mp4','飞书20220129-210653.mp4','飞书20220129-210549.mp4']
    output_name = 'out.mp4'
    output_path = os.path.join(data_root, output_name)

    audio = AudioFileClip('/mnt/bd/lwlq-workshop/yolov5/信天游/bgm.mp3')
    output_path_withaudio = os.path.join(data_root, output_name[:-4]+'_audio.mp4')


    video = VideoFileClip(output_path)# 读入视频

    # video = video.without_audio() 
    video = video.set_audio(audio)# 将音轨合成到视频中

    video.write_videofile(output_path_withaudio,audio_codec="aac")# 输出

    # #视频声音和背景音乐，音频叠加
    # audio_clip_add = CompositeAudioClip([video,audio])
    # #视频写入背景音
    # final_video = video_clip.set_audio(audio_clip_add)
    # #将处理完成的视频保存
    # final_video.write_videofile(output_path_withaudio)