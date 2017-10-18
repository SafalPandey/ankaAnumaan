import tensorflow as tf
import batch_maker,resize_data
import numpy as np
import cv2
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


n_classes = 10
keep_rate = 0.7

# keep_prob = tf.placeholder(tf.float32)

weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
'W_fc':tf.Variable(tf.random_normal([8*8*64,1024])),
'out':tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
'b_conv2':tf.Variable(tf.random_normal([64])),
'b_fc':tf.Variable(tf.random_normal([1024])),
'out':tf.Variable(tf.random_normal([n_classes]))}

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


saver = tf.train.Saver()
sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
saver.restore(sess,os.path.join(BASE_DIR,'models/CNN/353epochs.txt'))

x= tf.placeholder('float',[None,900])
def cnn_neural_network_model(x):

    x = tf.reshape(x, shape=[-1, 30, 30, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 8*8*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def hardcoded():
    prediction = cnn_neural_network_model([tf.cast(batch_maker.get_xy(2182,1)[0][0],tf.float32)])

    print(sess.run(tf.argmax(sess.run(tf.nn.softmax(sess.run(prediction))),1)))

def classify(arg,isImage = False):
    if isImage:
        img = resize_data.resize_img(arg,True)
    else:
        img = resize_data.resize_img(os.path.join(BASE_DIR+'/testing',img_path))
    x = np.array(img).flatten()
    prediction = cnn_neural_network_model([tf.cast(x,tf.float32)])
    # result = sess.run(tf.argmax(sess.run(tf.nn.softmax(sess.run(prediction))),1))
    # print(sess.run(tf.nn.softmax(prediction)))
    result = sess.run(tf.argmax(tf.nn.softmax(prediction),1))
    print(result)
    return result


if __name__ == "__main__":

    if  (len(sys.argv) == 1):
        hardcoded()

    else:
        img_path = sys.argv[1]
        print('Getting file ',img_path)
        # cv2.imshow('sad',cv2.imread(os.path.join(BASE_DIR,'testing/'+img_path)))
        # cv2.waitKey(0)
        classify(img_path)
