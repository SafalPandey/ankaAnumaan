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
batch_size = 500

x = tf.placeholder('float', [None, 900])
y = tf.placeholder('float')
# hidden_1_layer = {}
# hidden_2_layer = {}
# hidden_3_layer = {}
def neural_network_model(data):

    hidden_1_layer = {'weights':tf.get_variable('hl1_W',shape = [900, n_nodes_hl1], initializer = tf.zeros_initializer),
                      'biases':tf.get_variable('hl1_B',shape = [n_nodes_hl1],initializer = tf.zeros_initializer) }

    hidden_2_layer = {'weights':tf.get_variable('hl2_W',shape = [n_nodes_hl1, n_nodes_hl2],initializer = tf.zeros_initializer),
                      'biases':tf.get_variable('hl2_B',shape = [n_nodes_hl2],initializer = tf.zeros_initializer)}

    hidden_3_layer = {'weights':tf.get_variable('hl3_W',shape = [n_nodes_hl2, n_nodes_hl3],initializer = tf.zeros_initializer),
                      'biases':tf.get_variable('hl3_B',shape = [n_nodes_hl3],initializer = tf.zeros_initializer)}

    output_layer ={'weights':tf.get_variable('op_W',shape = [n_nodes_hl3, n_classes],initializer = tf.zeros_initializer),
                          'biases':tf.get_variable('op_B',shape = [n_classes],initializer = tf.zeros_initializer)}

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
    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(.01, global_step,100000, 0.96, staircase=True)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 600
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
            if((epoch+1) % 500 == 0):
                saver.save(sess,os.path.join(BASE_DIR,'models/'+str(epoch+1)+'epochs.txt'))

            if epoch_loss == 0:
                saver.save(sess,os.path.join(BASE_DIR,'models/'+str(epoch+1)+'epochs.txt'))
                break

        saver.save(sess,os.path.join(BASE_DIR,'models/'+str(epoch+1)+'epochs.txt'))
        # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #
        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
