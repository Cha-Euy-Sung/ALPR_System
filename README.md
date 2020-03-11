# ALPR(차량번호판인식)

딥러닝 기반의 Automatic License Plate Recognition 기술 구현 

[<img src="https://j.gifs.com/wVABrX.gif" width="70%">](https://youtu.be/EIZpI8A1Qe0)



## Preparation

##### Clone and install requirements
```
$ git clone https://github.com/Cha-Euy-Sung/ALPR
$ cd ALPR/
$ sudo pip3 install -r requirements.txt
```
##### download virtual environment(Optional)

[virt_env](https://drive.google.com/drive/folders/1qiPqo5hqrJK2ls1wVOfOg2Y41MnB3NOC?usp=sharing)
```
/home/pirl/yy 에 폴더 다운로드

yy/bin/ 경로에서 source ./activate 
```
##### download pretrained weights

[weights.zip](https://drive.google.com/file/d/1TVgXuKUXV57BzKNoc4lnE9514QN6Gh7-/view?usp=sharing)


##### sample data set for training

[baza_slika.zip](https://drive.google.com/file/d/1eTEZuuWt6ZiV22eOJ4NJYmcz914BwDpE/view?usp=sharing)



## Argument parser

|  -- |  type | default | help |
|:-----:|:-----:|:------:|:-----:|
|image_folder|str | /data/image/| path to image_folder which contains text images|
|batch_size|int|192|input batch size|
|img_size|int|800|size of each image dimension|
|video|str||only need when using video format|
|model_def|str|/config/custom.cfg|path to model definition file|
|weights_path|str|/weights/plate.weights|path to weights file|
|class_path|str|/data/custom/custom.names|path to class label file|
|conf_thres|str|0.8|object confidence threshold|
|n_cpu|int|8|number of cpu threads to use during batch generation|
|saved_model|str|/OCR/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth|path to saved_model to evaluation|
|Detection|str|Darknet|Detect plate stage. 'None' or 'Darknet'|
|Transformation|str|TPS|Transformation stage. 'None' or 'TPS'|
|FeatureExtraction|str|ResNet|FeatureExtraction stage. 'VGG'or'RCNN'or'ResNet'|
|Prediction|str|Attn|Prediction stage. 'CTC'or'Attn'|


## Test

```
python3 main.py 
```




## Cooperation

Thanks to Jinju@hhaahaha for implementing Optical Character Recognition, and working on this project together. 

## Credit
```
eriklindernoren/PyTorch-YOLOv3

tzutalin/labelImg

```

