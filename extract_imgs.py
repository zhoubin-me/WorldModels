import cv2
import numpy as np
import glob
from random import choice
import os

files = glob.glob('./record/*.npz')

imgs = np.load(choice(files))['sx']
print(len(imgs))
os.system('rm *.png')
for idx, img in enumerate(imgs):
    cv2.imwrite('out_%05d.png' % idx, img)
os.system('convert -delay 10 -loop 0 *.png data.gif')
os.system('scp data.gif bzhou@10.80.43.125:/home/bzhou/Dropbox')
os.system('rm *.png')
