from boundingBox import seg_crop
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

for f in os.listdir(os.path.join(BASE_DIR,'data')):
    if os.path.isfile(os.path.join(BASE_DIR+'/data',f)):
        seg_crop(f)
