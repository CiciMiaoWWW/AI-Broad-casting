import cv2
import argparse
import subprocess

args = argparse.ArgumentParser()
args.add_argument("--input", type=str, default="input.mp4", help='input video filename')
args.add_argument("--output", type=str, default="output.mp4", help='output video filename')
args.add_argument("--width", type=int, default=1920, help='output video width')
args.add_argument("--height", type=int, default=1080, help='output video height')
args = vars(args.parse_args())

video_name = args["input"]
cap = cv2.VideoCapture(video_name)
# fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("width:", width)
print("height:", height)
mid_name = "mid_" + args["output"]

cmd = ["ffmpeg"]
cmd.extend(["-i", video_name])
if height * args["width"] / width > args["height"]:
    cmd.extend(["-vf", "scale=-1:" + str(args["height"])])
else:
    cmd.extend(["-vf", "scale=" + str(args["width"]) + ":-1"])
cmd.extend(["-y", mid_name])
print(cmd)
subprocess.call(cmd)

cap = cv2.VideoCapture(mid_name)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x = int((args["width"] - width) / 2)
y = int((args["height"] - height) / 2)

cmd = ["ffmpeg"]
cmd.extend(["-i", mid_name])
cmd.extend(["-vf", "pad=" + str(args["width"]) + ":" + str(args["height"]) + ":" + str(x) + ":" + str(y) + ":black"])
cmd.extend(["-y", args["output"]])
subprocess.call(cmd)
print(cmd)
cmd = ["rm", mid_name]
subprocess.call(cmd)
#                 cmd.extend(["-i", fn])
#                 cmd.extend(["-codec", "copy"])

#                 if start_offset > 0:
#                     cmd.extend(["-ss", duration_to_hhmmss(start_offset)])
#                 if args.trim_end and duration > 0:
#                     cmd.extend(["-t", "%.3f" % duration])
#                 cmd.append(os.path.join(args.outdir, os.path.basename(fn)))
#                 print(cmd)
#                 check_call(cmd)