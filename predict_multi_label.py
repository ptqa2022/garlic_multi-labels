# -*- coding: utf-8 -*-
"""

Created by Pham Thi Quynh Anh, IAM, NARO
 
 Image classification of root-trimmed garlic using multi-label and multi-class classification with deep convolutional neural network

Pham Thi Quynh Anh1*, Dang Quoc Thuyet1,2, and Yuichi Kobayashi1 

1Institute of Agricultural Machinery, National Agriculture and Food Research Organization, 1-40-2 Nisshin, Kita-ku, Saitama City, Saitama 331-8537, Japan
2Current affiliation: Institute of Research and Development, Taisei Rotec Corporation, 1456, Kamiya Kounosu City, Saitama 365-0027, Japan
*Corresponding author: fumut476@affrc.go.jp
 
 Submitted Journal: Postharvest Biology and Technology
 
Usage
Note: pre-traing models are available at "models"
# 1 Edit parameter in the INPUT section 
 add image path
 img_path = "imgs/Garlic_01.jpg"
 
python predict_multi_label.py
 
Licence: MIT License
"""

############### INPUT ###################  
# add your image path here for prediction


img_path = "imgs/Garlic_01.jpg"
#img_path = "imgs/Garlic_02.jpg"
#img_path = "imgs/Garlic_03.jpg"






############ DONOT CHANGE THEM IF IT IS NOT NECESSARY #######################

import os
proba_min = 0.2
#second_model = True
second_model = False
time_stamp = False # write time stampe on the image
tag1  = "Multi-label model 1" #"With background class"
tag2  = "Multi-label model 2" #"Without background class"

img_width, img_height = 296, 296
img_width_show, img_height_show = 800, 600 
save_image = True 
viz_grad =  True 
viz_grad =  False
img_save_folder = "saveimages"
os.makedirs(img_save_folder, exist_ok = True )
#------------------


# the first multi-label model (with background)
model_path = "models/03_multi-label_bg/model_garlic_multi_labels_7classes.h5"
label_path = "models/03_multi-label_bg/label_garlic_multi_labels_7classes.classes"


# the second multi-label model (wtihout background)
model_path2 = "models/01_multi-label_non_bg/model_garlic_multi_labels_without_background_6classes.h5"
label_path2 = "models/01_multi-label_non_bg/mlb_garlic_multi_labels_without_background_6classes.classes"

# - ------ import necessary packages ----------

import tensorflow as tf
tf_ver = int(tf.__version__.split(".")[0])
if tf_ver==1: 
    tf1 = True
else: 
    tf1 = False 
if tf1:
    import keras
    from keras.models import load_model
    from keras.preprocessing import image
    from keras.layers.core import Lambda
    import keras.backend as K  
else:
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.layers import Lambda
    import tensorflow.keras.backend as K   
    from tensorflow.keras.utils import img_to_array
    tf.compat.v1.disable_eager_execution()

import numpy as np
import cv2
import glob 
import random 
import os
import datetime
import time
import imutils
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageFont, ImageDraw, Image

def predict_class(model,img_org, viz_grad = False):      
        width = np.size(img_org, 1)
        height = np.size(img_org, 0)
        img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img_org,(img_width, img_height)) 
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x1 = image.img_to_array(img1)
        x2 = np.expand_dims(x1, axis=0)
        images = x2/255.
        # predict   
        proba = model.predict(images)[0]
        idxs = np.argsort(proba)[::-1][:2]
        if proba_min>0 and proba_min <1:            
            idxs = [i for i in idxs if proba[i] > proba_min]

        if viz_grad:
            cam1, heatmap = grad_cam(model, images, "block5_conv3")       
            cam1 =0.4* np.float32(cam1) + np.float32(img_gray) #img_org cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            cam1 = 255 * cam1 / np.max(cam1) 
        else:
            #cam1 = x1 
            #cam1 = img
            cam1 = img_org
        return idxs,proba,cam1

def plot_label(img_org, viz_grad, save_image, path,img_save_folder): 
        width = np.size(img_org, 1)
        height = np.size(img_org, 0)
        # the first model
        idxs,proba,cam1 = predict_class(model,img_org,viz_grad)
        # loop over the indexes of the high confidence class labels
        label_save = ""
        for (i, j) in enumerate(idxs):
        # build the label and draw the label on the image
            label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
            if label_save == "":
                label_save = mlb.classes_[j]
            else:
                label_save = label_save +"_"+ mlb.classes_[j]
            cv2.putText(cam1, label, (10, (i * 30) + 25), #img_org cam1
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2) # 255

        # show the probabilities for each of the individual labels
        for (label, p) in zip(mlb.classes_, proba):
            print("{}: {:.2f}%".format(label, p * 100))
            
        if second_model:
            # the second model
            idxs2,proba2,cam12 = predict_class(model2, img_org,viz_grad)
            # loop over the indexes of the high confidence class labels
            label_save2 = ""
            for (i, j) in enumerate(idxs2):
            # build the label and draw the label on the image
                label2 = "{}: {:.2f}%".format(mlb2.classes_[j], proba2[j] * 100)
                if label_save2 == "":
                    label_save2 = mlb2.classes_[j]
                else:
                    label_save2 = label_save2 +"_"+ mlb2.classes_[j]
                cv2.putText(cam12, label2, (10, (i * 30) + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2) 

            # show the probabilities for each of the individual labels
            for (label2, p2) in zip(mlb2.classes_, proba2):
                print("{}: {:.2f}%".format(label2, p2 * 100))
            
        # draw the timestamp on the img_org
        timestamp = datetime.datetime.now()
        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        if time_stamp:
            cv2.putText(cam1, ts, (10, cam1.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,  # cam1 img_org
                0.4, (0, 0, 255), 1)
        cv2.putText(cam1, tag1, (10, cam1.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,  # cam1 img_org
            0.7, (255, 0, 0), 2)             
        
        if second_model:
            if time_stamp:
                cv2.putText(cam12, ts, (10, cam12.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,  # cam1 img_org
                    0.4, (0, 0, 255), 1)
            cv2.putText(cam12, tag2, (10, cam12.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,  # cam1 img_org
                0.7, (255, 0, 0), 2)                
        if not viz_grad:
            cv2.imshow("Evaluation. Press ESC for Escape",img_org)
        if viz_grad:
            #display = np.dstack((img_org, cam1))
            h1, w1 = img_org.shape[:2]
            h2, w2 = cam1.shape[:2]
            #create empty matrix
            if second_model:
                h3, w3 = cam12.shape[:2]
                vis = np.zeros((max(h1, h2, h3), w1+w2+w3,3), np.uint8)
                #combine 2 images
                vis[0:h1, 0:w1,:3] = img_org
                vis[0:h2, w1:w1+w2,:3] = cam1
                vis[0:h3, w1+w2:w1+w2+w3,:3] = cam12
            else:
                vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
                #combine 2 images
                vis[0:h1, 0:w1,:3] = img_org
                vis[0:h2, w1:w1+w2,:3] = cam1
            if save_image:
                img_name = img_save_folder +"/"+ label_save + "_" + path
                cv2.imwrite(img_name, vis)
            cv2.imshow("Evaluation. Press ESC for Escape",vis)
        elif save_image:
            img_name = img_save_folder +"/"+ label_save+ "_" + path
            cv2.imwrite(img_name, cam1)
            vis = cam1
            cv2.imshow("Evaluation. Press ESC for Escape",cam1)
        else:
            vis = cam1
            cv2.imshow("Evaluation. Press ESC for Escape",cam1)
        return(vis,label_save)
        

# vizualiation grad-cam
if viz_grad:
        def normalize(x):
            return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
            
        def grad_cam(input_model, image, layer_name):
            x = input_model.layers[0].get_layer(layer_name).output
            model1 = keras.models.Model(input_model.layers[0].layers[0].input, x)
            loss = K.sum(model1.get_layer(layer_name).output)
            conv_output = [l for l in model1.layers if l.name == layer_name][0].output          
            grads = normalize(K.gradients(loss, conv_output)[0])
            gradient_function = K.function([model1.layers[0].input], [conv_output, grads])
            output, grads_val = gradient_function([image])
            output, grads_val = output[0, :], grads_val[0, :, :, :]

            weights = np.mean(grads_val, axis = (0, 1))
            cam = np.ones(output.shape[0 : 2], dtype = np.float32)

            for i, w in enumerate(weights):
                cam += w * output[:, :, i]

            cam = cv2.resize(cam, (img_width_show, img_height_show))
            cam = np.maximum(cam, 0)
            heatmap = cam / np.max(cam)

            #Return to BGR 
            image = image[0, :]
            image -= np.min(image)
            image = np.minimum(image, 255)

            cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
            return np.uint8(cam), heatmap
 

# The first model 
# Load model and label_class

model = load_model(model_path)
mlb = pickle.loads(open(label_path, "rb").read())
#img_classes = mlb.classes_

# the second model
# Load model and label_class
if second_model:
    model2 = load_model(model_path2)
    mlb2 = pickle.loads(open(label_path2, "rb").read())
#img_classes2 = mlb2.classes_



# -- prediction function      
print(img_path)
img_name = os.path.basename(img_path)
img_org = cv2.imread(img_path,1)
img_org =  cv2.resize(img_org,(img_width_show, img_height_show ))
vis,label_save = plot_label(img_org,viz_grad, save_image, img_name,img_save_folder)
cv2.imshow("Evaluation",vis)
cv2.waitKey(0)


    

     