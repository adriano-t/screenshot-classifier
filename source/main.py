import subprocess

print("===========================")
print("== Screenshot classifier ==")
print("===========================")

##ARRAYS##
#array for neuralnetworks choise
nets = ["vgg16", "inception", "mobilenet","resnetv2", "nas","dense","vgg19"]
#array for crop's type (all= use all crops for that NN )
crops = ["full", "top-left", "top-right", "bottom-left", "bottom-right","all"]
#array for the chosen number of images to extract the features (max 100)
number = ["1", "5", "10", "20", "100"]



############ NET ###############
chosen = 0
i = 1
choices = "\n"
for name in nets:
    choices += str(i) + ") " + name + "\n" 
    i += 1
while chosen <= 0:
    chosen = input ("Choose a Neural Network: " + choices)
    try:
        chosen = int(chosen) 
        if chosen >= i or chosen <= 0:
            chosen = 0
            print("Insert a valid number.")
    except ValueError:
        print("Insert a valid number.")

net_name=nets[chosen-1]
print("--- Using " + net_name + " ---")



############ CROP ###############
chosen = 0
i = 1
choices = "\n"
for name in crops:
    choices += str(i) + ") " + name + "\n" 
    i += 1
while chosen <= 0:
    chosen = input ("Choose a crop method: " + choices)
    try:
        chosen = int(chosen) 
        if chosen >= i or chosen <= 0:
            chosen = 0
            print("Insert a valid number.") 
    except ValueError:
        print("Insert a valid number.") 
crop_select=crops[chosen-1]
print("Selecting " + crop_select)



############ NUMBER IMG ###############
chosen = 0
i = 1
choices = "\n"
for name in number:
    choices += str(i) + ") " + name + "\n" 
    i += 1
while chosen <= 0:
    chosen = input ("Choose the number of images to use for train : " + choices)
    try:
        chosen = int(chosen) 
        if chosen >= i or chosen <= 0:
            chosen = 0
            print("Insert a valid number.") 
    except ValueError:
        print("Insert a valid number.") 
number_train=number[chosen-1]
print("Selecting " + number_train + " images for training")



############ CORE ###############

#launch for the selected crop,NN and n_img extractor.py and train.py
if crop_select=="all":
# if i choose ALL, i need to launch all five crops
    for name in crops:
        if name== 'all':
            continue
        #arguments is a string that I pass to the others scripts
        arguments = net_name + " " + name+ " " + number_train
        print("\n### EXTRACTING ###")
        p = subprocess.Popen("python extractor.py " + arguments, shell=True)
        p.communicate()

        print("\n### TRAINING ###")
        p = subprocess.Popen("python train.py " + arguments, shell=True)
        p.communicate()

else:
    arguments = net_name + " " + crop_select+ " " + number_train
    print("\n### EXTRACTING ###")
    p = subprocess.Popen("python extractor.py " + arguments, shell=True)
    p.communicate()

    print("\n### TRAINING ###")
    p = subprocess.Popen("python train.py " + arguments, shell=True)
    p.communicate()



############ MODEL TESTING ###############

arguments = net_name + " " + crop_select+ " " + number_train
print("\n### TESTING ###")
p = subprocess.Popen("python classifier.py " + arguments, shell=True)
p.communicate()

print("### DONE ###")