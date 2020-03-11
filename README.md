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


##### download sample data

[baza_slika.zip](https://drive.google.com/file/d/1eTEZuuWt6ZiV22eOJ4NJYmcz914BwDpE/view?usp=sharing)


## Test

```
python3 detect.py --image_folder data/samples/ --weights_path weights/plate.weights 
```

##### Argument parser

    |   |  type | default | help |
    |-----|:-----:|:------:|:-----:|
    | --image_folder|str | data/image/| path to image_folder which contains text images|
    
    

## Credit
```
eriklindernoren/PyTorch-YOLOv3

tzutalin/labelImg

```

sudo apt install libdvdnav4 libdvdread4 gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly libdvd-pkg
sudo apt install ubuntu-restricted-extras
