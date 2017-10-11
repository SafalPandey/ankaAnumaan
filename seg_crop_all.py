from boundingBox import seg_crop
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def seg_crop_all():
    for f in os.listdir(os.path.join(BASE_DIR,'data')):
        if os.path.isfile(os.path.join(BASE_DIR+'/data',f)):
            print('Next file'+str(f))
            seg_crop(f)

if __name__ == "__main__":

    seg_crop_all()
