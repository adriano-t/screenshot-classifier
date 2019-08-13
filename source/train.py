from sklearn import svm
import numpy as np
import os

#FarCryNewDawn 2019-07-03 17-54-26-45.bmp

#https://scikit-learn.org/stable/modules/svm.html

features_path = "../features/"

feat_size = 25088
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
    labels = np.full((1, images_count), label)  

    #append labels
    all_labels = np.append(all_labels,  labels)
  
    label += 1
    
#todo
# qui abbiamo il matricione di features + labels
# dobbiamo fare il train della svm e salvare il modello in un file

#in futuro dobbiamo fare il train solo con parte dei dati, ad esempio 80%train e 20%test


#https://scikit-learn.org/stable/modules/svm.html




