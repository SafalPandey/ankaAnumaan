import os
import re
import shutil

from boundingBox import seg_crop
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def renameData():
    filenames = [f for f in os.listdir(os.path.join(BASE_DIR, 'croppedchildren'))
                 if os.path.isfile(os.path.join(BASE_DIR + '/croppedchildren', f))]

    newfilenames = []
    for f in filenames:
        fspl = f.split(' ')
        while len(re.sub('\D', '', fspl[3])) < 6:
            fspl[3] = '0' + fspl[3]
        newfilenames.append(' '.join(str(x) for x in fspl))

    for i, f in enumerate(filenames):
        if(re.sub('\D', '', f.split(' ')[3]) != 6):
            os.rename(os.path.join(BASE_DIR, 'croppedchildren/' + filenames[i]), os.path.join(
                BASE_DIR, 'croppedchildren/' + newfilenames[i]))
            # print('Renamed
            # '+os.path.join(BASE_DIR,'croppedchildren/'+filenames[i])+' to
            # '+newfilenames[i])

    newfilenames.sort()

    rowcount = 0
    colcount = 0

    for fname in newfilenames:
        if (rowcount <= 9):
            shutil.copy(os.path.join(BASE_DIR + "/croppedchildren", fname),
                        os.path.join(BASE_DIR, 'training_data/' + str(rowcount) + '.' + fname))
        else:
            shutil.copy(os.path.join(BASE_DIR + "/croppedchildren", fname),
                        os.path.join(BASE_DIR, 'test_data/' + str(rowcount) + '.' + fname))

        colcount += 1
        if colcount > 6:
            colcount = 0
            rowcount += 1

        if rowcount > 11:
            print(fname)
            rowcount = 0

if __name__ = "__main__":
    renameData()
