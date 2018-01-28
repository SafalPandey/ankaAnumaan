import os
import cv2
import numpy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resize_data():
    trainingData = [f for f in os.listdir(os.path.join(BASE_DIR, 'training_data')) if os.path.isfile(
        os.path.join(BASE_DIR + '/training_data', f))]

    # print(trainingData)

    for f in trainingData:
        img = cv2.imread(os.path.join(BASE_DIR, 'training_data/' + f))
        h, w, channels = img.shape
        if h < w:
            padded_img = cv2.copyMakeBorder(img, int((w - h) / 2), int(
                (w - h) / 2), 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            # cv2.imshow('img',padded_img)
        if w < h:

            padded_img = cv2.copyMakeBorder(img, 0, 0, int((h - w) / 2), int(
                (h - w) / 2), cv2.BORDER_CONSTANT, value=[255, 255, 255])

        resized_img = cv2.resize(
            padded_img, (30, 30), interpolation=cv2.INTER_AREA)
        rotated_img = numpy.rot90(resized_img, -1)
        cv2.imwrite(os.path.join(BASE_DIR, '30x30_data/' + f), rotated_img)


def resize_img(arg, isImage=False):
    if isImage:
        img = arg
    else:
        img = cv2.imread(arg)
    # print(path)
    h, w, channels = img.shape
    if h < w:
        padded_img = cv2.copyMakeBorder(img, int((w - h) / 2), int(
            (w - h) / 2), 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:

        padded_img = cv2.copyMakeBorder(img, 0, 0, int((h - w) / 2), int(
            (h - w) / 2), cv2.BORDER_CONSTANT, value=[255, 255, 255])

    resized_img = cv2.resize(
        padded_img, (30, 30), interpolation=cv2.INTER_AREA)
    # cv2.imshow('img',resized_img)
    # cv2.waitKey(0)
    # print(resized_img.shape)
    # rotated_img = numpy.rot90(resized_img,-1)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    return resized_img

if __name__ == "__main__":
    resize_data()
