# garlic_multi-labels
multi-label model for garlic root trimming image classification

# This is code for the following paper.

Image classification of root-trimmed garlic using multi-label and multi-class classification with deep convolutional neural network

Pham Thi Quynh Anh1*, Dang Quoc Thuyet1,2, and Yuichi Kobayashi1 

1Institute of Agricultural Machinery, National Agriculture and Food Research Organization, 1-40-2 Nisshin, Kita-ku, Saitama City, Saitama 331-8537, Japan
2Current affiliation: Institute of Research and Development, Taisei Rotec Corporation, 1456, Kamiya Kounosu City, Saitama 365-0027, Japan

*Corresponding author: fumut476@affrc.go.jp

Journal: Postharvest Biology and Technology
https://www.journals.elsevier.com/postharvest-biology-and-technology


If you use or find out our data useful, please cite our paper in the journal of Postharvest Biology and Technology
Citation:
Anh, P. T. Q., Thuyet, D. Q. & Kobayashi, Y. Image classification of root-trimmed garlic using multi-label and multi-class classification with deep convolutional neural network. Postharvest Biol. Technol. 190, 111956, 2022.
https://doi.org/10.1016/j.postharvbio.2022.111956

## Updated
5-9-2022
add online paper link and citation

3-17-2022
- upload the code


## Dependencies
- python 3.7 
- numpy
- scipy
- opencv-python
- sklearn
- tensorflow 2x (tested 2.7) or 1x (tested 1.9)
- keras
- pickle
- imutils
- matplotlib
- seaborn
- pandas

## Quick Start


0. Check all dependencies installed

1. download dataset at
    
    Dataset
    https://drive.google.com/file/d/1zXM80R2ziObzo5DnQpKlGtH2S3ENAqj8/view?usp=sharing

    Dataset licence: Creative Commons Attribution-NonCommercial-ShareAlike (CC-BY-NC-SA)
    http://creativecommons.org/licenses/by-nc-sa/4.0/

2. Train the model

 1- Edit parameters in the INPUT section in the file  multi_train.py

     Traing the model with multi_lable =>   multi_label = True
     Traing the model with multi_class =>   multi_label = False

     Traing the model with backround image =>   background_class = True
     Traing the model without backround image =>   background_class = False

  Note "dataset" folder should be in the same root folder with multi_train.py.
 
  2- Run the training
 
    python multi_train.py


3 . Predict image class

 1-download pretrained models if necessary
    
    Pre-train weights
    https://drive.google.com/file/d/1875xGUcKhDV4izvr4-ptBD--cIaU0QNG/view?usp=sharing

    Pre-trained weight licence:Creative Commons Attribution-NonCommercial-ShareAlike (CC-BY-NC-SA)
    http://creativecommons.org/licenses/by-nc-sa/4.0/

    save pre-trained weights to "models" folder

 2- Edit parameters in the INPUT section  of predict_multi_label.py
    
    for example: add image path
    img_path = "imgs/Garlic_01.jpg"
 
 3- Run
    
    python predict_multi_label.py
