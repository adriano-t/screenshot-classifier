
import os
import time
import numpy as np
from keras.preprocessing import image
from sklearn import svm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

start = time.time()

crop = True
if (crop):
    crop_w = int(1920 / 5)
    crop_h = int(1080 / 3)
    crop_x = crop_w * 4
    crop_y = crop_h * 2

# vgg16 / inception / mobilenet
net_name = "mobilenet" 

if(net_name == "vgg16"):
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input as preprocess_vgg
    modelClass = VGG16
    preprocess_function = preprocess_vgg
    feat_size = 25088

if(net_name == "inception"):
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_v3 import preprocess_input as preprocess_inception
    modelClass = InceptionV3
    preprocess_function = preprocess_inception
    feat_size = 51200

if(net_name == "mobilenet"):
    from keras.applications.mobilenet_v2 import MobileNetV2
    from keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
    modelClass = MobileNetV2
    preprocess_function = preprocess_mobilenet
    feat_size = 62720

with open('../models/labels_' + net_name + '.txt') as f:
    labels_names = [line.strip() for line in f]


print("Extracting features Using "+net_name+" net")
extractor_model = modelClass(weights='imagenet', include_top=False)
print(labels_names)
test_path = "../test/"
correct = 0
count = 0
y_true = []
y_pred = []
print("Loading model " + net_name)
model = joblib.load('../models/'+net_name+'.model')
for dirname in os.listdir(test_path):
    for fname in os.listdir(test_path + dirname):

        #skip unused directories
        if not (dirname + ".np") in labels_names:
            continue

        #load target image
        img_path = test_path + dirname + "/" + fname 
        #img_path = "/home/empo/Scrivania/progettottr/dset/overwatch/Overwatch 2019-06-01 09-33-21-84.bmp"
        img = image.load_img(img_path)
        if (crop):
            img = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        img = img.resize((224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_function(img_data)
        
        features = extractor_model.predict(img_data).flatten() 

        label_truth = labels_names.index(dirname + ".np")
        label_predicted = model.predict(np.asmatrix(features)) 
        
        print("probability: " + str(model.predict_proba(np.asmatrix(features)))) 
         
        y_true.append(label_truth) 
        y_pred.append(label_predicted)
        if(label_predicted == label_truth):
            correct += 1
            print("[v] " + fname + " | predicted: " + str(labels_names[int(label_predicted)])) 
        else :
            print("[ ] " + fname + " | predicted: " + str(labels_names[int(label_predicted)])) 
        count += 1

print("ratio: " + str((correct / count) * 100) + "%")


##########################
#### CONFUSION MATRIX ####
##########################
color_map=plt.cm.Blues 
np.set_printoptions(precision=2)

cm = confusion_matrix(y_true, y_pred)
#classes = classes[unique_labels(y_true, y_pred)] # Only use the labels that appear in the data 
print('Confusion matrix, without normalization')
print(cm)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=color_map)
ax.figure.colorbar(im, ax=ax)
# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=labels_names, yticklabels=labels_names,
        title= 'Confusion matrix ' + net_name,
        ylabel='True label',
        xlabel='Predicted label')


# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

plt.show()

print("Execution time:" + str(time.time() - start)) 