
import os
from PIL import Image


dataset_path = "../dset/"
for game_dir in os.listdir(dataset_path):

    image_list = os.listdir(dataset_path + game_dir)
    for image_name in image_list:
        name_ext = os.path.splitext(image_name)
        ext = name_ext[1]

        if ext == ".bmp":
            print("converting" + image_name)
            new_name = name_ext[0] + ".jpg"
            full_path = dataset_path + game_dir + "/" + image_name
            out_path = dataset_path + game_dir + "/" + new_name

            #convert
            img = Image.open(full_path)
            img.save(out_path, 'jpeg')
            os.remove(full_path)
        
        