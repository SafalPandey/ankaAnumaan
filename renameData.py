from boundingBox import seg_crop
import os
import shutil
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


filenames = [f for f in os.listdir(os.path.join(BASE_DIR,'croppedchildren'))    if os.path.isfile(os.path.join(BASE_DIR+'/croppedchildren',f))]

newfilenames=[]
for f in filenames:
    fspl = f.split(' ')
    while len(re.sub('\D','',fspl[3])) < 6:
        fspl[3] = '0'+fspl[3]
    newfilenames.append(' '.join(str(x) for x in fspl))

print(len(filenames))
print(len(newfilenames))

for i,f in enumerate(filenames):
    if(re.sub('\D','',f.split(' ')[3]) != 6):
        os.rename(os.path.join(BASE_DIR,'croppedchildren/'+filenames[i]),os.path.join(BASE_DIR,'croppedchildren/'+newfilenames[i]))
        # print('Renamed '+os.path.join(BASE_DIR,'croppedchildren/'+filenames[i])+' to '+newfilenames[i])

newfilenames.sort()

rowcount = 0
colcount = 0

for fname in newfilenames:
    print(fname)
    if (rowcount <=9):
        shutil.copy(os.path.join(BASE_DIR+"/croppedchildren", fname),os.path.join(BASE_DIR,'training_data/'+str(rowcount)+'.'+fname))
    colcount +=1
    if colcount > 6:
        colcount = 0
        rowcount +=1

    if rowcount > 11:
        rowcount = 0
