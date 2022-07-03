# An Improved Event-Independent Network for Polyphonic Sound Event Localization and Detection
An Improved Event-Independent Network (EIN) for Polyphonic Sound Event Localization and Detection (SELD)

from Centre for Vision, Speech and Signal Processing, University of Surrey.


## Usage
wth
the model pth weights should stored under`out_train/ein_seld/EINV2_tPIT_n1/checkpoints`

Hyper-parameters are stored in `./configs/ein_seld/seld.yaml`. You can change some of them, such as `train_chunklen_sec`, `train_hoplen_sec`, `test_chunklen_sec`, `test_hoplen_sec`, `batch_size`, `lr` and others.

For unit test use during inference time:

```bash
  # first cut video based on the duration length and translate them into wav format # and move it to "./_dataset/dataset_root/foa_eval"
  ffmpeg -ss $x -i video//string_cut.mp4 -vn -acodec pcm_s16le -ar 24000 -ac 2 -t 60 "./_dataset/dataset_root/foa_eval/string_left_$x.wav"

  # Norm Audio
  # process the audio in and stored it in `_dataset/dataset_root/foa_eval/normed`
  python seld/NormAudio.py -c $CONFIG_FILE infer --num_workers=0

  # Extract data
  # take audio from `_dataset/dataset_root/foa_eval/normed`, extract features, and store it into `_hdf5/dcase2020task3/scalar`
  python seld/main.py -c $CONFIG_FILE preprocess --preproc_mode='extract_data' --dataset_type='eval'

  # predict
  # take feature from `_hdf5/dcase2020task3/scalar` and infer the timestamp features.
  # feature is stored under a npy file
  python seld/main.py -c $CONFIG_FILE infer --num_workers=0 --npy_file_name='model_results'

  # post process the time stamps features and get the time stamps results. the results are stored under '.\timestamp_results'
  python investigate.py --filename='result_string' --outputfilename=$x
```



The bash script below will automate the **whole** timestamp output procedure (upper parts) given the file in video/video.mp4:

```bash
bash scripts/realtime.sh
```

## Reference

1. Archontis Politis, Sharath Adavanne, and Tuomas Virtanen. A dataset of reverberant spatial sound scenes with moving sources for sound event localization and detection. In Proceedings of the Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE2020). November 2020. [URL](https://arxiv.org/abs/2006.01919)

2. Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, and Tuomas Virtanen. Joint measurement of localization and detection of sound events. In IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). New Paltz, NY, Oct 2019. [URL](https://ieeexplore.ieee.org/abstract/document/8937220?casa_token=Z4aGA4E2Dz4AAAAA:BELmzMjaZslLDf1EN1NVZ92_9J0PRnRymY360j--954Un9jb_WXbvLSDhp--7yOeXp0HXYoKuUek)

3. Sharath Adavanne, Archontis Politis, Joonas Nikunen, and Tuomas Virtanen. Sound event localization and detection of overlapping sources using convolutional recurrent neural networks. IEEE Journal of Selected Topics in Signal Processing, 13(1):34–48, March 2018. [URL](https://ieeexplore.ieee.org/abstract/document/8567942)

4. https://github.com/yinkalario/DCASE2019-TASK3

