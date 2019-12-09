#C:/Users/adriano/AppData/Local/Programs/Python/Python35/python.exe ./heatmap.py mobilenet full
import os
import sys
import time
import math
import numpy as np
from netinfo import *
from datetime import datetime
from keras.preprocessing import image
from sklearn import svm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

start = time.time()

net_name = sys.argv[1]
crop_mode = sys.argv[2]

crop_modes = { 
    "bottom-right":  (4/5, 2/3, 1, 1), 
    "top-left":  (0, 0, 1/5, 1/3),
    "top-right": (0, 2/3, 1/5, 1),
    "bottom-left": (0, 2/3, 1/5, 1), 
}

# Check if the model is trained
if not os.path.exists('../models/labels_' + net_name + '.txt') or not os.path.exists('../models/'+net_name+ "_" + crop_mode+'.model'):
    print("Error: there is no trained model for " + net_name)

# Create report file
if not os.path.exists("../reports/"):
    os.mkdir("../reports/")
    #%m/%d/%Y, %H:%M:%S
freport_name = "../reports/" + net_name + "_" + crop_mode + "_report_{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
freport = open(freport_name + ".txt", "w+")
freport.write(net_name + " " + crop_mode + "\n")
freport.write("FAILED:\n\n")

if(net_name == "vgg16"):
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input as preprocess_vgg
    modelClass = VGG16
    preprocess_function = preprocess_vgg 

if(net_name == "inception"):
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_v3 import preprocess_input as preprocess_inception
    modelClass = InceptionV3
    preprocess_function = preprocess_inception 

if(net_name == "mobilenet"):
    from keras.applications.mobilenet_v2 import MobileNetV2
    from keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
    modelClass = MobileNetV2
    preprocess_function = preprocess_mobilenet 

if(net_name == "resnetv2"):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_resnetv2
    modelClass = InceptionResNetV2
    preprocess_function = preprocess_resnetv2 
    
if(net_name == "nas"):
    from keras.applications.nasnet import NASNetLarge
    from keras.applications.nasnet import preprocess_input as preprocess_nas
    modelClass = NASNetLarge
    preprocess_function = preprocess_nas
    
if(net_name == "dense"):
    from keras.applications.densenet import DenseNet201
    from keras.applications.densenet import preprocess_input as preprocess_dense
    modelClass = DenseNet201
    preprocess_function = preprocess_dense
    
if(net_name == "vgg19"):
    from keras.applications.vgg19 import VGG19
    from keras.applications.vgg19 import preprocess_input as preprocess_vgg19
    modelClass = VGG19
    preprocess_function = preprocess_vgg19

feat_size = features_sizes[net_name]
input_size = input_sizes[net_name]

test_path = "../heatmap/"
out_path = "../heatmap_results/"
if not os.path.exists(out_path):
    os.mkdir(out_path)
correct = 0
count = 0
y_true = []
y_pred = []    

#python heatmap.py mobilenet full

with open('../models/labels_' + net_name+ '.txt') as f:
    labels_names = [os.path.splitext(line.strip().split('$')[0])[0] for line in f]
print(labels_names)

print("Extracting features Using "+net_name +" net")
extractor_model = modelClass(weights='imagenet', include_top=False)

print("Loading model " + net_name)
print('../models/'+net_name+ '_' + crop_mode +'.model')

model = joblib.load('../models/'+net_name+ '_' + crop_mode +'.model')
for dirname in os.listdir(test_path):
    for fname in os.listdir(test_path + dirname):
 
        #skip unused directories
        if not (dirname) in labels_names:
            continue

        print("Generating heatmap for " + dirname + "/" + fname )

        #load target image
        img_path = test_path + dirname + "/" + fname 
        img = image.load_img(img_path)
    
        if(crop_mode != "full"): 
            mode = crop_modes[crop_mode]
            crop_x1 = mode[0]
            crop_y1 = mode[1]
            crop_x2 = mode[2]
            crop_y2 = mode[3]
            img = img.crop((crop_x1 * img.width, crop_y1 * img.height, crop_x2 * img.width, crop_y2 * img.height,)) 
            
        img = img.resize(input_size)
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_function(img_data)
        
        features = extractor_model.predict(img_data).flatten() 
        label_truth = labels_names.index(dirname)
        prob_array = model.predict_proba(np.asmatrix(features))
        probability_full = prob_array[0][label_truth]
        
        print("probability: " + str(probability_full)) 
        # mean_color = (0,0,0)
        # for i in range(img.size[0]):    # for every col:
        #     for j in range(img.size[1]):    # For every row
        #         mean_color = tuple(map(lambda x, y: x + y,mean_color, img[i,j])) #sum tuple
        
        #calculate mean color
        I = np.array(img) 
        mean_color = np.mean(I, axis=(0, 1))
        print(mean_color)

        #create sliding box
        box_scale = (8, 5)
        step = 10
        box_size = (math.floor(img.size[0]/box_scale[0]), math.floor(img.size[1]/box_scale[1]), 3)
        #print(box_size) 
        #print(mask)
        
        heat = np.zeros(img.size)
        # slide the mask
        for x in range(0, img.size[0] - 1, step): 
            for y in range(0, img.size[1] - 1, step):                
                I = np.array(img)
                I[x : min(img.size[0], x + box_size[0]), y : min(img.size[1], y + box_size[1]), :] = mean_color
        
                # convert back to img
                #img = image.array_to_img(I) #maybe np.uint8(I) 
                #img.show()

                # preprocess np array
                I = np.expand_dims(I, axis=0)
                I = preprocess_function(I) 

                # extract features
                features = extractor_model.predict(I).flatten()  
                label_truth = labels_names.index(dirname)
                prob_array = model.predict_proba(np.asmatrix(features))
                probability = prob_array[0][label_truth]

                # calculate probability delta
                delta = probability - probability_full
                heat[x:x+step, y:y+step] = delta
                print("delta("+str(x) +","+str(y)+") = " + str(delta))
        

        heat *= 255.0 / heat.max()
        #plt.imshow(heat, cmap='hot', interpolation='nearest')
        plt.figure(1)
        plt.subplot(211)
        plt.imshow(img) 
        plt.subplot(212) 
        plt.imshow(heat, cmap='bwr', interpolation='bilinear')
        #plt.show() 
        if not os.path.exists(out_path + dirname ):
            os.mkdir(out_path + dirname )
        
        out_name = os.path.splitext(fname)[0] + ".jpg"
        plt.savefig(out_path + dirname + "/"+ out_name, bbox_inches='tight', pad_inches=0)
        count += 1

print("ratio: " + str((correct / count) * 100) + "%")
freport.write("\n----------------------\n\nRatio: " + str((correct / count) * 100) + "%")

print("Execution time:" + str(time.time() - start))