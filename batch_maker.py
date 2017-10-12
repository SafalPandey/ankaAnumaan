import os
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

filenames = [f for f in os.listdir(os.path.join(BASE_DIR,'30x30_data')) if os.path.isfile(os.path.join(BASE_DIR+'/30x30_data',f))]
one_hot = { 0:[1,0,0,0,0,0,0,0,0,0],
            1:[0,1,0,0,0,0,0,0,0,0],
            2:[0,0,1,0,0,0,0,0,0,0],
            3:[0,0,0,1,0,0,0,0,0,0],
            4:[0,0,0,0,1,0,0,0,0,0],
            5:[0,0,0,0,0,1,0,0,0,0],
            6:[0,0,0,0,0,0,1,0,0,0],
            7:[0,0,0,0,0,0,0,1,0,0],
            8:[0,0,0,0,0,0,0,0,1,0],
            9:[0,0,0,0,0,0,0,0,0,1]    }

def get_xy(offset,batch_size):
    x = []
    labels = []
    for f in filenames[offset*batch_size : batch_size*(offset+1)]:
        img = cv2.imread(os.path.join(BASE_DIR+'/30x30_data',f),cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        img = np.array(img).flatten()
        x.append(img)
        labels.append([])

        labels[-1].append(one_hot[int(f[0])])


    return x,labels


if __name__ == "__main__":
    total = 0
    i = 0
    while(1):
        x , y = getnames(i,100)
        total += len(y)

        print(i)
        i+=1
        if len(x) == 0:
            break

    print(total)
