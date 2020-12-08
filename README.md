# SB-assignment2

## Ear Detection Using Haar Cascades

### Requirements
For the code in jupyter notebooks to be runnable you need to create a conda environment from the `environment.yml` file.
Additionally if you want to train the cascades on your own you need to download the [OpenCV applications](https://sourceforge.net/projects/opencvlibrary/files/3.4.12/) and move 
`opencv_createsamples.exe` and `opencv_traincascade.exe` in your project folder.
You need to get all the images:
- a set of grayscale images from [Kaggle](https://www.kaggle.com/muhammadkhalid/negative-images), save them to folder *neg_img_gs*,
- *2017 Train/Val annotations* from [COCO website](https://cocodataset.org/#download), save them to folder *COCO/annotations*. 
Conda installation doesn't work on Windows, so `pycocotools` are not included in the `environment.yml`. You need to run `pip install pycocotools-windows` if you use Windows 
or `conda install -c conda-forge pycocotools` otherwise.
- Positive and test images are available [here](http://awe.fri.uni-lj.si/). You need to save them to folder *AWEForSegmentation*.

### Training
Data preparation is described and implemented in `data_preparation.ipynb`. The outputs needed for training are the `.txt` description files for negative images and those images. 
Bellow are the commands for *OpenCV* applications:
- `pos.vec` file is an output of command `opencv_createsamples.exe -info info.dat -w 24 -h 24 -num 1000 -vec pos.vec`
- cascades are saved in folders named *cascade_*. Words after _ describe how the cascade was trained. Bellow we have multiple commands how to get the cascades that are included in 
the repository in folders with names as written in argument -data:
  - `opencv_traincascade.exe -data cascade_basic/ -vec pos.vec -bg bg.txt -w 24 -h 24 -numPos 800 -numNeg 1700 -numStages 10`
  - `opencv_traincascade.exe -data cascade_basic_all/ -vec pos.vec -bg bg.txt -w 24 -h 24 -numPos 800 -numNeg 1700 -numStages 10 -mode ALL`
  - `opencv_traincascade.exe -data cascade_basic_maxFAR03/ -vec pos.vec -bg bg.txt -w 24 -h 24 -numPos 800 -numNeg 1700 -numStages 10 -maxFalseAlarmRate 0.3`
  - `opencv_traincascade.exe -data cascade_basic_all_maxFAR03/ -vec pos.vec -bg bg.txt -w 24 -h 24 -numPos 800 -numNeg 1700 -numStages 10 -mode ALL -maxFalseAlarmRate 0.3`
  - `opencv_traincascade.exe -data cascade_gs/ -vec pos.vec -bg negative_gs.txt -w 24 -h 24 -numPos 800 -numNeg 3000 -numStages 10`
  - `opencv_traincascade.exe -data cascade_gs_all/ -vec pos.vec -bg negative_gs.txt -w 24 -h 24 -numPos 800 -numNeg 3000 -numStages 10 -mode ALL`
  - `opencv_traincascade.exe -data cascade_gs_maxFAR03/ -vec pos.vec -bg negative_gs.txt -w 24 -h 24 -numPos 800 -numNeg 3000 -numStages 10 -maxFalseAlarmRate 0.3`
  - `opencv_traincascade.exe -data cascade_gs_all_maxFAR03/ -vec pos.vec -bg negative_gs.txt -w 24 -h 24 -numPos 800 -numNeg 3000 -numStages 10 -mode ALL -maxFalseAlarmRate 0.3`
  - `opencv_traincascade.exe -data cascade_coco/ -vec pos.vec -bg negative_coco.txt -w 24 -h 24 -numPos 800 -numNeg 3000 -numStages 10`
  - `opencv_traincascade.exe -data cascade_coco_people/ -vec pos.vec -bg negative_coco_people.txt -w 24 -h 24 -numPos 800 -numNeg 4500 -numStages 10`
  - `opencv_traincascade.exe -data cascade_coco_people_all_maxFAR03/ -vec pos.vec -bg negative_coco_people.txt -w 24 -h 24 -numPos 800 -numNeg 4500 -numStages 10 -mode ALL -maxFalseAlarmRate 0.3`
 
### Models
All the trained cascades are available in folders *cascade_*. I propose you use `cascade_coco_people_all_maxFAR03`, 
from my testing it has the best results. For detection choose parameters `scaleFactor` 1.1 and `minNeighbours` 6.
You can read more in the report `ZrimsekUrsa2.pdf`.
You can check how they work live with function `camera_test()` in `testing.ipynb` or use them for detecting ears on images with support of `cv2` python library.
