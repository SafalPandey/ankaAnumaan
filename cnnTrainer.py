import tensorflow as tf
import batch_maker as input_data
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

n_classes = 10
batch_size = 20

x = tf.placeholder('float', [None,30,30])
y = tf.placeholder('float')

# keep_rate = 0.7
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([8*8*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 30, 30, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 8*8*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    # fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 1000
    saver = tf.train.Saver()
    count = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(len(input_data.filenames)/batch_size)+1):
                epoch_x, epoch_y = input_data.get_xy(i,batch_size,isCNN=True)
                # print(len(epoch_y))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            if (epoch+1) % 500 == 0 :
                saver.save(sess,os.path.join(BASE_DIR,'models/CNN/'+str(epoch+1)+'epochs.txt'))
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            # if epoch_loss < 10000:
            #     saver.save(sess,os.path.join(BASE_DIR,'models/CNN/withLoss'+str(epoch_loss)+'in'+str(epoch+1)+'epochs.txt'))
            #     k = input('Want to Stop? [y/n]')
            #     if k == 'y':
            #         break
            if epoch_loss == 0:
                count +=1
                if count == 10:
                    saver.save(sess,os.path.join(BASE_DIR,'models/CNN/'+str(epoch+1)+'epochs.txt'))
                    break
        saver.save(sess,os.path.join(BASE_DIR,'models/CNN/'+str(hm_epochs)+'epochs.txt'))

train_neural_network(x)
