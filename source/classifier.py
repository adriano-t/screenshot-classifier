from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_vgg
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_inception
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from sklearn import svm
import numpy as np
import os
import time
from sklearn.externals import joblib

start = time.time()


# vgg16 / inception / mobilenet
net_name = "vgg16" 

if(net_name == "vgg16"):
    modelClass = VGG16
    preprocess_function = preprocess_vgg
    feat_size = 25088

if(net_name == "inception"):
    net_name = "inception"
    modelClass = InceptionV3
    preprocess_function = preprocess_inception
    feat_size = 51200

if(net_name == "mobilenet"):
    net_name = "mobilenet"
    modelClass = MobileNetV2
    preprocess_function = preprocess_mobilenet
    feat_size = 62720



with open('../models/labels_' + net_name + '.txt') as f:
    labels_names = [line.strip().split(',') for line in f]

print("Extracting features Using "+net_name+" net")
extractor_model = modelClass(weights='imagenet', include_top=False)
print(labels_names)
test_path = "../test/"
correct = 0
count = 0

print("Loading model " + net_name)
model = joblib.load('../models/'+net_name+'.model')
for dirname in os.listdir(test_path):
    for fname in os.listdir(test_path + dirname):
        #load target image
        img_path = test_path + dirname + "/" + fname 
        #img_path = "/home/empo/Scrivania/progettottr/dset/overwatch/Overwatch 2019-06-01 09-33-21-84.bmp"
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_function(img_data)
        
        features = extractor_model.predict(img_data).flatten() 

        label = model.predict(np.asmatrix(features)) 
        if(str(labels_names[int(label)]) == "['" + dirname +".np']"):
            correct += 1
            print("[v] " + fname + " | predicted: " + str(labels_names[int(label)])) 
        else :
            print("[ ] " + fname + " | predicted: " + str(labels_names[int(label)])) 
        count += 1

print("ratio: " + str((correct / count) * 100) + "%")

print("Execution time:" + str(time.time() - start)) 