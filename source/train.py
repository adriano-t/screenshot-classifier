
import os 
import sys
import time
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from netinfo import * # file containing features sizes and input sizes for all NN

#https://scikit-learn.org/stable/modules/svm.html

#chosen NN and crop mode
net_name = sys.argv[1]
crop_mode = sys.argv[2]
#check label.txt if exists
label_path='../models/labels_' + net_name + '.txt'
if os.path.exists(label_path):
    exit
#from netinfo
feat_size = features_sizes[net_name]
#if not exists models,create it!
if not os.path.exists('../models/'):
    os.mkdir('../models/')
#load labels
flabels=open(label_path, 'w')

start = time.time()

features_path = "../features/" + net_name+"/"+crop_mode+"/"
all_labels =  np.array([])
all_features = np.empty((0, feat_size), float)
print("Loading features")
label = 0

for fname in sorted(os.listdir(features_path)):
    print("- " + fname) 
    #load all the features of the game
    f = open(features_path + fname, "rb")
    features = np.load(f) 
    f.close()

    #append features
    all_features = np.append(all_features,  features, axis=0)

    #create labels
    images_count =  features.shape[0]
    print("images: " + str(images_count))
    labels = np.full((1, images_count), label)  
    #append labels
    all_labels = np.append(all_labels,  labels)
    #write label to file
    game_name=fname[:fname.index('.np')]

    flabels.write("%s\n" % game_name) 

    label += 1

flabels.close()

print("Training model: " + str(all_features.shape))
model = svm.SVC(gamma='scale', probability=True)
model.fit(all_features, all_labels) 

#save
joblib.dump(model, '../models/' + net_name + "_" + crop_mode + '.model')

print("Execution time:" + str(time.time() - start))