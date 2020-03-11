# ALPR(차량번호판인식)
## Introduction
 Automatic License Plate Recognition based on Deep learning. 
 - Work on both image and video format
 - Models were trained by datasets which contains Eng/Number. 
   (Unfortunately, it doesn't recognize the Korean yet)
 - Used YOLOv3 for license plate detection
 - OCR based on CNN,RNN,TPS and Attn|CTC
 
## Demo
[<img src="https://j.gifs.com/wVABrX.gif" width="70%">](https://youtu.be/EIZpI8A1Qe0)



## Preparation

#### Clone and install requirements
```
$ git clone https://github.com/Cha-Euy-Sung/ALPR
$ cd ALPR/
$ sudo pip3 install -r requirements.txt
```
#### env
```
| UBUNTU 16.04 | python 3.6 | pytorch 1.4.0 | Opencv-python 4.2 | CUDA 10.1 |
```
#### download virtual environment(Optional)

- [virt_env](https://drive.google.com/drive/folders/1qiPqo5hqrJK2ls1wVOfOg2Y41MnB3NOC?usp=sharing)

download 'yy.zip' at /home/pirl/yy 
```
$ cd yy
$ cd bin
$ source ./activate 
```
#### download pretrained weights

- [weights.zip](https://drive.google.com/file/d/1TVgXuKUXV57BzKNoc4lnE9514QN6Gh7-/view?usp=sharing)


#### sample data set used for training

- [baza_slika.zip](https://drive.google.com/file/d/1eTEZuuWt6ZiV22eOJ4NJYmcz914BwDpE/view?usp=sharing)



# Training OCR model
## Train OCR model before put in Plate Recognition model
### Clone and install requirements
```
$ mkdir OCR
$ cd OCR
$ git clone https://github.com/clovaai/deep-text-recognition-benchmark.git
```
## Getting Started
### Dependency
- This work was tested with PyTorch 1.3.1, CUDA 10.1, python 3.6 and Ubuntu 16.04. <br> You may need `pip3 install torch==1.3.1`. <br>
In the paper, expriments were performed with **PyTorch 0.4.1, CUDA 9.0**.
- requirements : lmdb, pillow, torchvision, nltk, natsort
```
pip3 install lmdb pillow torchvision nltk natsort
```

### Download lmdb dataset for traininig and evaluation from [here](https://drive.google.com/drive/folders/192UfE9agQUMNq6AgU3_E05_FcPZK4hyt)
data_lmdb_release.zip contains below. <br>
training datasets : [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)[1] and [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)[2] \
validation datasets : the union of the training sets [IC13](http://rrc.cvc.uab.es/?ch=2)[3], [IC15](http://rrc.cvc.uab.es/?ch=4)[4], [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[5], and [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[6].\
evaluation datasets : benchmark evaluation datasets, consist of [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[5], [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[6], [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions)[7], [IC13](http://rrc.cvc.uab.es/?ch=2)[3], [IC15](http://rrc.cvc.uab.es/?ch=4)[4], [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf)[8], and [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html)[9].

### Run demo with pretrained model
1. Download pretrained model from [here](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW)
2. Add image files to test into `demo_image/`
3. Run demo.py (add `--sensitive` option if you use case-sensitive model)
```
CUDA_VISIBLE_DEVICES=0 python3 demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder demo_image/ \
--saved_model TPS-ResNet-BiLSTM-Attn.pth
```
### Arguments
* `--train_data`: folder path to training lmdb dataset.
* `--valid_data`: folder path to validation lmdb dataset.
* `--eval_data`: folder path to evaluation (with test.py) lmdb dataset.
* `--select_data`: select training data. default is MJ-ST, which means MJ and ST used as training data.
* `--batch_ratio`: assign ratio for each selected data in the batch. default is 0.5-0.5, which means 50% of the batch is filled with MJ and the other 50% of the batch is filled ST.
* `--data_filtering_off`: skip [data filtering](https://github.com/clovaai/deep-text-recognition-benchmark/blob/f2c54ae2a4cc787a0f5859e9fdd0e399812c76a3/dataset.py#L126-L146) when creating LmdbDataset. 
* `--Transformation`: select Transformation module [None | TPS].
* `--FeatureExtraction`: select FeatureExtraction module [VGG | RCNN | ResNet].
* `--SequenceModeling`: select SequenceModeling module [None | BiLSTM].
* `--Prediction`: select Prediction module [CTC | Attn].
* `--saved_model`: assign saved model to evaluation.
* `--benchmark_all_eval`: evaluate with 10 evaluation dataset versions, same with Table 1 in our paper.
* `--FT`: fine tunning model with custom dataset.

### Make Custom Datasets
1. make gt.txt for create 
```
pip3 install fire
python3 make_gt_txt.py --file_name gt.txt --dir_path data/image/
```
2. Create your own lmdb dataset.
```
python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/
```
At this time, `gt.txt` should be `{imagepath}\t{label}\n` <br>
For example
```
test/word_1.png Tiredness
test/word_2.png kills
test/word_3.png A
...
```
3. Modify `--select_data`, `--batch_ratio`, and `opt.character`, see [this issue](https://github.com/clovaai/deep-text-recognition-benchmark/issues/85).

### After FT model with custom dataset
put the 'best_model.sh' into './PlateRecognition/OCR/saved_models'











## Test

```
python3 main.py 
```
#### Argument parser

|  -- |  <font color="blue">type | default | help </font>|
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




## Cooperation

Thanks to Jinju@hhaahaha for implementing Optical Character Recognition, and working on this project together. 

## Credit
```
eriklindernoren/PyTorch-YOLOv3

tzutalin/labelImg

```

