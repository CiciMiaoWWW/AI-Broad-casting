import cv2, os
from moviepy import *
from moviepy.editor import *
import numpy as np
import imageio
import argparse

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", required=True, type=str, default="out.mp4", help='input video filename, output will be [input_name]_audio.mp4')
    ap.add_argument("--bgm", required=True, type=str, help='bgm filename,can be mp3, mp4, wav etc')
    args = vars(ap.parse_args())

    video = VideoFileClip(args["vid"])# 读入视频
    audio = AudioFileClip(args["bgm"])
    output_path_withaudio = args["vid"][:-4]+'_audio.mp4'
    
    video = video.set_audio(audio)# 将音轨合成到视频中
    video.write_videofile(output_path_withaudio,audio_codec="aac")# 输出