import cv2, os
from moviepy import *
from moviepy.editor import *
import numpy as np
import imageio
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type = str, default = '/mnt/bd/lwlq-workshop/yolov5/信天游/', help = 'data root')
    parser.add_argument('--output', type = str, default = 'out11.mp4', help = 'output name')
    parser.add_argument('--audio', type = str, default = 'bgm.mp3')
    parser.add_argument('--video', type = str, default = 'out11_audio.mp4', help = 'output name')


    opt = parser.parse_args()

    print(opt)
    #data_root = '/mnt/bd/lwlq-workshop/yolov5/信天游/'
    #output_name = 'out11.mp4'
    output_path = os.path.join(opt.dataroot, opt.output)
    audio_path = os.path.join(opt.dataroot, opt.audio)
    video_path = os.path.join(opt.dataroot, opt.video)
    audio = AudioFileClip(audio_path)
    #output_path_withaudio = os.path.join(data_root, output_name[:-4]+'_audio.mp4')
    video = VideoFileClip(video_path)# 读入视频
    # video = video.without_audio() 
    video = video.set_audio(audio)# 将音轨合成到视频中
    video.write_videofile(output_path,audio_codec="aac")# 输出

    # #视频声音和背景音乐，音频叠加
    # audio_clip_add = CompositeAudioClip([video,audio])
    # #视频写入背景音
    # final_video = video_clip.set_audio(audio_clip_add)
    # #将处理完成的视频保存
    # final_video.write_videofile(output_path_withaudio)