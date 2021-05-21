# APGAN (Image-to-image task) -Tensorflow
This project is used for the testing of APGAN (not yet released).

### Result
<div align="center">
  <img src="./result/out.gif">
</div>

<div align="center">
  <img src="city_game.mp4">
</div>

## Requirements
* python3.7
* numpy
* opencv-python
* tensorflow-gpu

## Description
You can use this project to complete part of the domain transformation of APGAN(APGAN: Adaptive Penalty Technique to Improve Image-to-Image Generation Quality).APGAN is a multi-scale generation architecture, but we currently only use a single-layer generator size in `net.py`, which is sufficient to produce good results. The entire architecture will be updated later, and then we are not opening the training process for the time being. The full `ortrain.py` and `net.py` will be updated later

## Preparation
* **Cityscape**
  * Download from [here]( https://www.cityscapes-dataset.com), We used the leftImg8bit file in Cityscape for domain transformation with GTA
* **GTA**
  * Download from [here]( https://download.visinf.tu-darmstadt.de/data/from_games)
* **Horse2zebra**
  *  you can refer to CycleGAN for this part
  
## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── Domain A
           ├── 000001.jpg 
           ├── 000002.png
           └── ...
       ├── Domain B
           ├── 000001.jpg
           ├── 000002.png
           └── ...
       ├── test A
           ├── a.jpg 
           ├── b.png
           └── ...
       ├── test B (unsupervised)
           ├── a.jpg 
           ├── b.png
           └── ...
```

## Tseting
In initialization.py, parameter `test_data_path` is your test data path, parameter `snapshot_dir` is your cheakpoint path, then set the `name` parameter to test. After setting，run main.py, You can get the result of the domain transformation.

## Weight file
We upload weights for Cityscape image segmentation, you can download from [here]( https://pan.baidu.com/s/13aKwSvzepVBsAHZESpFWtA)，Web extraction code：data

## Acknowledgement
Thanks to Yanshuai Wang for his contribution to the dataset GTA
