import cv2, os
from moviepy import *
from moviepy.editor import *
import numpy as np
import imageio

if __name__ == '__main__':
    data_root = '/mnt/bd/lwlq-workshop/yolov5/信天游/'
    output_name = 'out11.mp4'
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