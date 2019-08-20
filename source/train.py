from sklearn import svm
import numpy as np
import os 
from sklearn.externals import joblib
import time

#https://scikit-learn.org/stable/modules/svm.html


# vgg16 / inception / mobilenet
net_name = "mobilenet" 

if(net_name == "vgg16"):
    feat_size = 25088

if(net_name == "inception"):
    feat_size = 51200

if(net_name == "mobilenet"):
    feat_size = 62720



flabels=open('../models/labels_' + net_name + '.txt', 'w')
start = time.time()

features_path = "../features/" + net_name+"/"
 
all_labels =  np.array([])
all_features = np.empty((0, feat_size), float)

print("Loading features")
label = 0
for fname in os.listdir(features_path):
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
    flabels.write("%s\n" % fname) 

    label += 1

flabels.close()
        
#TRAINING SVM
#support vector classification SVC 
# gamma='auto' uses 1 / n_features
# gamma='scale' uses 1 / (n_features * X.var()) as value of gamma.

print("Training model: " + str(all_features.shape))
model = svm.SVC(gamma='scale', probability=True)
model.fit(all_features, all_labels) 

#save
joblib.dump(model, '../models/' + net_name + '.model')

print("Execution time:" + str(time.time() - start))
 
#todo
# qui abbiamo il matricione di features + labels
# dobbiamo fare il train della svm e salvare il modello in un file

#in futuro dobbiamo fare il train solo con parte dei dati, ad esempio 80%train e 20%test


#https://scikit-learn.org/stable/modules/svm.html




