import os
from PIL import Image

#timer
import time
start_time = time.time()

directory = os.getcwd()
# read a image and crop
crop_box = (0,70,800,420) #(left, upper, right, lower)

#actual code
for filename in os.listdir(directory):

#for testing
#im = Image.open("20161101_154630.jpg")

#one tab right in actual
    if filename.endswith(".jpg") & ~(filename.endswith("_crop.jpg")):
        im = Image.open(filename)
        fn = filename[:-4]
        I = im.crop(crop_box)
        I = I.resize((80, 35), Image.ANTIALIAS)
        I.save('%s_crop.jpg'%fn)
        os.remove(filename)

#timer
print("--- %s seconds ---" % (time.time() - start_time))
