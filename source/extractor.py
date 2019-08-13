from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_vgg
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_inception
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
import numpy as np
import os


dataset_path = "../dset/"
features_path = "../features/"
feat_size = 25088

print("Using VGG16 net")
model = VGG16(weights='imagenet', include_top=False)
#model.summary()

#for takes all images on directories
for game_dir in os.listdir(dataset_path):
    out_path = features_path + "/" + game_dir  + ".np"
    if(os.path.exists(out_path)):
        print(game_dir + " already exists")
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
            img_data = preprocess_vgg(img_data)
            
            #extract features
            vgg16_feature = model.predict(img_data)

            #add row to the features matrix
            features = np.asmatrix(vgg16_feature.flatten())
            all_features = np.append(all_features, np.asmatrix(features), axis=0)
            
            i += 1
        except:
            print("exception: skipped")
    
    np.save(feat_file,all_features)
    feat_file.close()
    print("Saved to " + feat_file.name)
    



'''
# Inizializzo il modello, preaddestrato su un enorme dataset di immagini
# include_top = False significa che voglio usare il modello 
# come un estrattore di feature e non come un classificatore
model = VGG16(weights='imagenet', include_top=False)
#model.summary() # stampa la struttura del modello

# Ogni modello riscala l'immagine ad una dimensione prefissata:
# Nel caso di rete VGG, 224x224
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
# i valori RGB dell'immagine vengono standardizzati
img_data = preprocess_vgg(img_data)

# Estrazione feature
vgg16_feature = model.predict(img_data)
print("VGG NET")
print(vgg16_feature.flatten().shape)

# Stesso discorso se voglio cambiare rete neurale
model = InceptionV3(weights='imagenet', include_top=False)
#model.summary() # stampa la struttura del modello

img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
# i valori RGB dell'immagine vengono standardizzati
img_data = preprocess_inception(img_data)

inception_feature = model.predict(img_data)
print("INCEPTION NET")
print(inception_feature.flatten().shape)

# Terza rete: mobilenet, spesso usata per applicazioni mobile
model = MobileNetV2(weights='imagenet', include_top=False)

img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
# i valori RGB dell'immagine vengono standardizzati
img_data = preprocess_mobilenet(img_data)

mobilenet_feature = model.predict(img_data)
print("MOBILE NET")
print(mobilenet_feature.flatten().shape)

# Per un elenco delle possibili reti, si veda
# la tabella "Documentation for individual models" su
# https://keras.io/applications/'''