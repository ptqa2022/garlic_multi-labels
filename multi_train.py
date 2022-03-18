# Created by Pham Thi Quynh Anh, IAM , NARO
"""
Image classification of root-trimmed garlic using multi-label and multi-class classification with deep convolutional neural network

Pham Thi Quynh Anh1*, Dang Quoc Thuyet1,2, and Yuichi Kobayashi1 

1Institute of Agricultural Machinery, National Agriculture and Food Research Organization, 1-40-2 Nisshin, Kita-ku, Saitama City, Saitama 331-8537, Japan
2Current affiliation: Institute of Research and Development, Taisei Rotec Corporation, 1456, Kamiya Kounosu City, Saitama 365-0027, Japan

*Corresponding author: fumut476@affrc.go.jp
Submitted Journal: Postharvest Biology and Technology

Usage

############### how to use code################
# 1 Edit parameter in the INPUT section 

 Traing the model with multi_lable =>   multi_label = True
 Traing the model with multi_class =>   multi_label = False

 Traing the model with backround image =>   background_class = True
 Traing the model without backround image =>   background_class = False

# Note "dataset" folder should be in the same root folder with multi_train.py
# 2 Run the training
python multi_train.py

Licence: MIT License
"""

############### INPUT ###################  


 
multi_label = True
#multi_label = False
 
background_class = True
#background_class = False



############ DONOT CHANGE THEM IF IT IS NOT NECESSARY #######################

#brightness_range =  None
brightness_range = (0.8,1.2)

# Hyperparameters
dropout_rate = 0.2
seed_no1=123
seed_no2=42
EPOCHS1 = 40
EPOCHS2 = 12
INIT_LR1 = 1e-4
INIT_LR2 = 1e-5
BS = 32
height = 296 
width = 296
depth = 3

import numpy as np

import tensorflow as tf
tf_ver = int(tf.__version__.split(".")[0])
if tf_ver==1:
    tf1 = True
else: 
    tf1 = False 
    
if tf1:
    import keras
    from keras.models import Sequential
    from keras.layers.core import Activation, Flatten, Dropout,Dense
    from keras.preprocessing.image import ImageDataGenerator, img_to_array
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam
    from keras.applications import VGG16
else:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense, BatchNormalization
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import img_to_array
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.applications.vgg16 import VGG16
    tf.compat.v1.disable_eager_execution()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if multi_label:
    from sklearn.preprocessing import MultiLabelBinarizer  
    from sklearn.metrics import multilabel_confusion_matrix
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support
    from sklearn.metrics import roc_auc_score, hamming_loss, jaccard_score , log_loss, zero_one_loss
else: 
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay

import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from imutils import paths
import random
import pickle
import os
import time
import datetime
import sys


def list_files(base_path, valid_exts=None, contains=None, keep_folder = None):
    image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for (root_dir, dir_names, filenames) in os.walk(base_path):
        for filename in filenames:    
            if contains is not None and filename.find(contains) == -1:
                continue
            ext = filename[filename.rfind("."):].lower()
            if valid_exts is None or ext.endswith(image_types):
                if keep_folder is None:
                    image_path = os.path.join(root_dir, filename)
                    yield image_path
                else:
                    if root_dir.lower().find(keep_folder.lower())  == -1:
                        image_path = os.path.join(root_dir, filename)
                        yield image_path
                        
def plot_history_all(H,N_epoch, plot_name = "Test_fig.png", plot_save = False, tf1 = False):
    if tf1:
        label_acc = "acc"
        label_val = 'val_acc'
    else:
        label_acc = "accuracy"
        label_val = 'val_accuracy'
        
    plt.style.use("ggplot")
    plt.figure()
    N = N_epoch
    plt.plot(np.arange(0, N), H["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H[label_acc], label="train_acc")
    plt.plot(np.arange(0, N), H[label_val], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss, Accuracy")
    plt.ylim((0,1))
    plt.legend(loc="right")
    if plot_save:
        plt.savefig(plot_name)
    else:    
        plt.show()




        
data_folder_name = "dataset"

if multi_label:
    if background_class:
        model_name = "model_garlic_multi_labels"
        label_name = "label_garlic_multi_labels"
        plot_name = "plot_garlic_multi_labels"
    else:
        model_name = "model_garlic_multi_labels_without_background"
        label_name = "mlb_garlic_multi_labels_without_background"
        plot_name = "plot_garlic_multi_labels_without_background"
else:
    if background_class:
        model_name = "model_garlic_multi_classes"
        label_name = "label_garlic_multi_classes"
        plot_name = "plot_garlic_multi_classes"    
    else:
        model_name = "model_garlic_multi_classes_without_background"
        label_name = "label_garlic_multi_classes_without_background"
        plot_name = "plot_garlic_multi_classes_without_background"    


# Loading image paths 
if background_class:
    image_paths = sorted(list(list_files(data_folder_name)))
else:
    image_paths = sorted(list(list_files(data_folder_name,keep_folder = "background")))

random.seed(seed_no1) 
random.shuffle(image_paths)

data = []
labels_str = []
for image_path in image_paths:
    img = cv2.imread(image_path,1)
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    data.append(img)
    if multi_label:
        label = image_path.split(os.path.sep)[-2].split("_")
        labels_str.append(label)
    else:    
        label = image_path.split(os.path.sep)[-2]
        labels_str.append(label)
    
data = np.array(data, dtype="float") 
labels_str = np.array(labels_str)

#Class labels
if multi_label:
    mlb = MultiLabelBinarizer()
else:
    mlb = LabelBinarizer()

labels = mlb.fit_transform(labels_str)    

# class labels
for (i, label) in enumerate(mlb.classes_):
	print(f"{i + 1}. {label}")

(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size = 0.3, random_state = seed_no2)  #42  123 0.2

train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1, 
        shear_range=0.2, 
        zoom_range=0.2,
	    horizontal_flip=True, 
        brightness_range=brightness_range, 
        fill_mode="nearest")
        
test_datagen = ImageDataGenerator(rescale=1.0/255.0)  

#Model structure 
n_classes=len(mlb.classes_)
if multi_label:
    final_act_fc="sigmoid"
else:
    final_act_fc="softmax" 

# load the VGG16 network 
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(height, width, depth))
           
#start transferring 
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate)) 
model.add(Dense(n_classes, activation=final_act_fc)) 

conv_base.trainable = False

if tf1:
    opt1 = Adam(lr = INIT_LR1, decay = INIT_LR1 / EPOCHS1)
else:
    opt1 = Adam(learning_rate = INIT_LR1, decay = INIT_LR1 / EPOCHS1)

if multi_label:
    model.compile(loss="binary_crossentropy", optimizer=opt1, metrics=["accuracy"])
else:
    model.compile(loss="categorical_crossentropy", optimizer=opt1, metrics=["accuracy"])
    
# train the network
if tf1:
    H1 = model.fit_generator(
    	train_datagen.flow(trainX, trainY, batch_size = BS),
        validation_data = test_datagen.flow(testX, testY, batch_size = BS),
    	steps_per_epoch = len(trainX) // BS,
    	epochs = EPOCHS1, verbose = 1)
else:
    H1 = model.fit(
    	train_datagen.flow(trainX, trainY, batch_size = BS),
        validation_data = test_datagen.flow(testX, testY, batch_size = BS),
    	steps_per_epoch = len(trainX) // BS,
    	epochs = EPOCHS1, verbose = 1)

# Fine tunning
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block3_conv1': 
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False  

if tf1:
    opt2 = Adam(lr=INIT_LR2, decay=INIT_LR2 / EPOCHS2)
else:
    opt2 = Adam(learning_rate=INIT_LR2, decay=INIT_LR2 / EPOCHS2)

if multi_label:
    model.compile(loss="binary_crossentropy", optimizer=opt2, metrics=["accuracy"])
else:
    model.compile(loss="categorical_crossentropy", optimizer=opt2, metrics=["accuracy"])    

model.summary()

# re-training
if tf1:
    H2 = model.fit_generator(
    	train_datagen.flow(trainX, trainY, batch_size=BS),
    	#validation_data=(testX, testY),
        validation_data = test_datagen.flow(testX, testY, batch_size = BS),
    	steps_per_epoch=len(trainX) // BS,
    	epochs=EPOCHS2, verbose=1)
else:
    H2 = model.fit(
    	train_datagen.flow(trainX, trainY, batch_size=BS),
    	#validation_data=(testX, testY),
        validation_data = test_datagen.flow(testX, testY, batch_size = BS),
    	steps_per_epoch=len(trainX) // BS,
    	epochs=EPOCHS2, verbose=1)        
        
# save the model to disk

model_name = model_name + "_{}classes".format(n_classes)  + ".h5"

if tf1:
    model.save(model_name)
else:
    model.save(model_name,save_format="h5")


# save the multi-label binarizer to disk
model_label_name = label_name +"_{}classes".format(n_classes) + ".classes"
f = open(model_label_name, "wb")
f.write(pickle.dumps(mlb))
f.close()

df1 = pd.DataFrame(H1.history)
df2 = pd.DataFrame(H2.history)

H = pd.concat([df1,df2],ignore_index=True)  
csv_name = "H_all_Transfer_finetune_history" + ".csv"
H.reset_index(drop=True)
H.to_csv(csv_name)


# plot the training loss and accuracy
plot_name = plot_name +  "_{}classes".format(n_classes) + ".png"

N = EPOCHS1 + EPOCHS2

# plot 1 same figure
plot_history_all(H,N_epoch = N, plot_name = plot_name, plot_save = True, tf1 = tf1)

# evaluate network on validation data set
validation_loss, validation_acc = model.evaluate(test_datagen.flow(testX,testY, batch_size=BS,shuffle=False))
print("\nValidation loss: {}, validation accuracy: {}".format(validation_loss, 100*validation_acc))

# evaluate network on train dateset
train_loss, train_acc = model.evaluate(test_datagen.flow(trainX, trainY, batch_size=BS,shuffle=False))
print("\nTrain loss: {}, train accuracy: {}".format(train_loss, 100*train_acc))

# predict
if tf1:
    test_pred = model.predict_generator(test_datagen.flow(testX,testY, batch_size=BS,shuffle=False),)
else:    
    test_pred = model.predict(test_datagen.flow(testX,testY, batch_size=BS,shuffle=False),)

img_classes =  mlb.classes_

if multi_label: 
    preds_test = np.where(test_pred< 0.5, 0, 1)
    mcm = multilabel_confusion_matrix(testY, preds_test)
    print('Confusion Matrix\n')
    print(mcm)    
else: 
    preds_test  = np.where(test_pred< 0.5, 0, 1)
    preds_test_1d= np.argmax(test_pred, axis = 1)
    testY_1d = np.argmax(testY, axis = 1)
    cm = confusion_matrix(testY_1d, preds_test_1d)
    print('Confusion Matrix\n')
    print(cm)
    
# report as table
if multi_label:
    report0= classification_report(testY, preds_test,target_names=img_classes)
    report1= classification_report(testY, preds_test,target_names=img_classes,output_dict=True)
else:
    report0= classification_report(testY_1d, preds_test_1d,target_names= img_classes)
    print(report0)
    report1= classification_report(testY_1d, preds_test_1d,target_names=img_classes,output_dict=True)   

print('\nClassification Report\n')
print(report0)

# saving csv file
clsf_report =pd.DataFrame(report1)
clsf_report.to_csv('Conv3_Classification_report_test.csv', index= True)


