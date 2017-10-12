import tensorflow as tf
import batch_maker,resize_data
import numpy as np
import cv2
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10

hidden_1_layer = {'weights':tf.get_variable('hl1_W',shape = [900, n_nodes_hl1]),
                  'biases':tf.get_variable('hl1_B',shape = [n_nodes_hl1]) }

hidden_2_layer = {'weights':tf.get_variable('hl2_W',shape = [n_nodes_hl1, n_nodes_hl2]),
                  'biases':tf.get_variable('hl2_B',shape = [n_nodes_hl2])}

hidden_3_layer = {'weights':tf.get_variable('hl3_W',shape = [n_nodes_hl2, n_nodes_hl3]),
                  'biases':tf.get_variable('hl3_B',shape = [n_nodes_hl3])}

output_layer ={'weights':tf.get_variable('op_W',shape = [n_nodes_hl3, n_classes]),
                      'biases':tf.get_variable('op_B',shape = [n_classes])}

saver = tf.train.Saver()
sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
saver.restore(sess,os.path.join(BASE_DIR,'models/100epochs.txt'))

x= tf.placeholder('float',[None,900])
def neural_network_model(data):


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def hardcoded():
    prediction = neural_network_model([tf.cast(batch_maker.get_xy(2182,1)[0][0],tf.float32)])

    print(sess.run(tf.argmax(sess.run(tf.nn.softmax(sess.run(prediction))),1)))

def classify(arg,isImage = False):
    if isImage:
        img = resize_data.resize_img(arg,True)
    else:
        img = resize_data.resize_img(os.path.join(BASE_DIR+'/testing',img_path))
    x = np.array(img).flatten()
    prediction = neural_network_model([tf.cast(x,tf.float32)])
    result = sess.run(tf.argmax(sess.run(tf.nn.softmax(sess.run(prediction))),1))
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
