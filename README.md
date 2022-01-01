# Basic Code
## 1. Use The Code
Configuration Environment
```bash
conda create -n code_base python=3.6
conda activate code_base
pip install -r requirements.txt -i  https://pypi.doubanio.com/simple/
sh env.sh
```
Train
```bash
python train.pycd
```
Test
```bash
python test.py
```
## 2. Reconstruction Effect
Left: source image | Right: reconstructed image

<img src="./test_input_folder/xi.jpg" width="256" height="256" />       <img src="./test_results/xi.jpg" width="256" height="256" />