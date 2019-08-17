from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_vgg
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_inception
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
import numpy as np
import os
import time

start = time.time()
dataset_path = "../dset/"
features_path = "../features/"
if(not os.path.exists(features_path)):
    os.mkdir(features_path)


# vgg16 / inception / mobilenet
net_name = "vgg16" 

if(net_name == "vgg16"):
    modelClass = VGG16
    preprocess_function = preprocess_vgg
    feat_size = 25088

if(net_name == "inception"):
    modelClass = InceptionV3
    preprocess_function = preprocess_inception
    feat_size = 51200

if(net_name == "mobilenet"):
    modelClass = MobileNetV2
    preprocess_function = preprocess_mobilenet
    feat_size = 62720




print("\n\n")
print("===== Using " + net_name + " =====")
model = modelClass(weights='imagenet', include_top=False) #model.summary() 
print("\n\n")
#create directory if not exists
features_dir = features_path + net_name + "/"
if(not os.path.exists(features_dir)):
    os.mkdir(features_dir)

#for takes all images on directories
for game_dir in os.listdir(dataset_path):

    out_path =  features_dir  + game_dir  + ".np"

    if(os.path.exists(out_path)):
        print(game_dir + ": game features already extracted")
        continue

    feat_file = open(out_path, "wb+")
    all_features = np.empty((0, feat_size), float)
    
    #execute vgg16 on each image in the directory
    i = 0
    image_list = os.listdir(dataset_path + game_dir)
    print("Extracting: " + game_dir + " (" + str(len(image_list)) + " images)")
    for image_name in image_list:
        
        try:
            #resize images for being used on vgg16
            print(str(i) + ") " + image_name)
            img_path = dataset_path + game_dir + "/" + image_name
            img = image.load_img(img_path, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_function(img_data)
            
            #extract features
            features =  model.predict(img_data).flatten() 

            #add row to the features matrix 
            all_features = np.append(all_features, np.asmatrix(features), axis=0)
            
            i += 1
            if(i > 100):
                break
        except Exception as e: print(e + ":skipped")
    
    np.save(feat_file,all_features)
    feat_file.close()
    print("Saved to " + feat_file.name)
    


print("Execution time:" + str(time.time() - start))


'''
# Per un elenco delle possibili reti, si veda
# la tabella "Documentation for individual models" su
# https://keras.io/applications/
# '''