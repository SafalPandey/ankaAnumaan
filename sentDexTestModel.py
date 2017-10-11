import tensorflow as tf
import batch_maker as input_data
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# from tensorflow.examples.tutorials.mnist import input_data as mnist_data
# mnist =  mnist_data.read_data_sets("/tmp/data/", one_hot = True)


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 900])
y = tf.placeholder('float')
# hidden_1_layer = {}
# hidden_2_layer = {}
# hidden_3_layer = {}
def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([900, n_nodes_hl1]),name = 'hl1_W'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]),name = 'hl1_B')}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]),name = 'hl2_W'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]),name = 'hl2_B')}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]),name = 'hl3_W'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]),name = 'hl3_B')}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes]),name = 'op_W'),
                    'biases':tf.Variable(tf.random_normal([n_classes]),name = 'op_B')}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 100
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(len(input_data.filenames)/batch_size)):
                epoch_x, epoch_y = input_data.get_xy(i,batch_size)
                # print(len(epoch_y))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

        saver.save(sess,os.path.join(BASE_DIR,'models/'+hm_epochs+'epochs.txt'))
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
