import os
import time
import pyscreenshot as ImageGrab 
from datetime import datetime
gameName = "pes19"
dirname = "../" + gameName + "/"
if(not os.path.exists(dirname)):
    os.mkdir(dirname)

while True:
    im = ImageGrab.grab()
    dt = datetime.now()
    fname = dirname + gameName + "_{}_{}.png".format(dt.strftime("%H_%M_%S"), dt.microsecond // 100000)
    im.save(fname, 'png')
    time.sleep(10)