import subprocess

print("===========================")
print("== Screenshot classifier")
print("===========================")
nets = ["vgg16", "inception", "mobilenet","resnetv2", "nas","dense","resnet","vgg19","resnext" ]

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
crops = ["full", "bottom-right", "top-left" ]
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

arguments = net_name + " " + crop_select
print("\n### EXTRACTING ###")
p = subprocess.Popen("python extractor.py " + arguments, shell=True)
p.communicate()

print("\n### TRAINING ###")
p = subprocess.Popen("python train.py " + arguments, shell=True)
p.communicate()

print("\n### TESTING ###")
p = subprocess.Popen("python classifier.py " + arguments, shell=True)
p.communicate()


print("### DONE ###")