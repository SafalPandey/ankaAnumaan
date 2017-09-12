from boundingBox import seg_crop
import os

for f in os.listdir('D:/_nepali char/data'):
    if os.path.isfile(os.path.join('D:/_nepali char/data',f)):
        seg_crop(f)
