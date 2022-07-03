#!/bin/bash

set -e
CONFIG_FILE='./configs/ein_seld/seld.yaml'


# variables section
x=0
total_dur=0
dir=



# need to have a function to cut the video into one minute interval
# get the video length
video_length=$(ffprobe -i video//string_cut.mp4 -show_entries format=duration -v quiet -of csv="p=0")
echo "Video length is $video_length"


# change to int
INT_video_length=$(printf "%.f" "$video_length")
echo $INT_video_length

# this loop take in one video for every 1 min
# need to make it as a function ( with time argument so that I can save the timestamp file as log)
# begin while loop
# while x less than video length # x=current video starting time
while [ $x -le $INT_video_length ]
do

  start=`date +%s`

  # wait one minute then start
#  sleep 1m

  # first cut video based on the duration length and translate them into wav format # and move it to "./_dataset/dataset_root/foa_eval"
  ffmpeg -ss $x -i video//string_cut.mp4 -vn -acodec pcm_s16le -ar 24000 -ac 2 -t 60 "./_dataset/dataset_root/foa_eval/string_left_$x.wav"

  # Norm Audio
  python seld/NormAudio.py -c $CONFIG_FILE infer --num_workers=0

  # Extract data
  python seld/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_data' --dataset_type='eval'

  # predict
  python seld/main.py -c $CONFIG_FILE infer --num_workers=0 --npy_file_name='model_results'

  # post process and get the time stamps
  python investigate.py --filename='result_string' --outputfilename=$x


  # delete the wav file to save storage
  filename="./_dataset/dataset_root/foa_eval/string_left_$x.wav"
  # Check the file is exists or not
  if [ -f $filename ]; then
     rm $filename
     echo "$filename is removed"
  fi
  # delete end

  # end while loop ( in each loop: ~1 min 15 second)
  echo "Loop $x operation is done"
  x=$(( $x + 60 ))


  # time
  end=`date +%s`
  runtime=$((end-start))
  echo "each loop take $runtime second"

done

