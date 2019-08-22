import os
import shutil

    
if os.path.exists("../features") and  input("Delete features? y/n: ") == "y":
    shutil.rmtree("../features")

if os.path.exists("../models") and input("Delete models? y/n: ")=="y":
    shutil.rmtree("../models")

print("done")