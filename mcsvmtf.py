import csv
import extractfeatures as ef
import matplotlib.pyplot as plt
import numpy as np
import sourcedata as sd
import tensorflow as tf


"""
The multiclass SVM is based on
https://github.com/nfmcclure/tensorflow_cookbook/blob/master/
04_Support_Vector_Machines/06_Implementing_Multiclass_SVMs/
06_multiclass_svm.ipynb
"""

# parameters to be varied
eeg_epoch_width_in_s = 4
pe_orders = list(range(3, 8))
pe_delays = list(range(1, 11))


def read_features(filename, dataset):
    with open(filename, newline='') as csvfile:
        filereader = csv.reader(csvfile)
        for row in filereader:
            data = [float(i) for i in row]
            dataset.append(data)
    return

# read dataset - 1 fold only for now
# TODO: do all folds
fold = 0
pe_path = 'data/pe/'
template = '{0}_ew{1}_f{2}_{3}t{4}_{5}t{6}.csv'
common_labels = [str(eeg_epoch_width_in_s), str(fold), str(pe_orders[0]),
                 str(pe_orders[-1]), str(pe_delays[0]), str(pe_delays[-1])]
train_epochs = []
train_labels = []
test_epochs = []
test_labels = []
input_filename = template.format('train_data', *common_labels)
read_features(pe_path + input_filename, train_epochs)
input_filename = template.format('train_labels', *common_labels)
read_features(pe_path + input_filename, train_labels)
input_filename = template.format('test_data', *common_labels)
read_features(pe_path + input_filename, test_epochs)
input_filename = template.format('test_labels', *common_labels)
read_features(pe_path + input_filename, test_labels)


def create_labels_array(labels_1d, num_classes):
    labels_list = []
    for i in range(num_classes):
        labels = np.array([1 if label == i else -1 for label in labels_1d])
        labels_list.append(labels)
    return np.array(labels_list)

# convert datasets to numpy arrays
num_classes = 6
train_epochs = np.array(train_epochs)
train_labels = create_labels_array(train_labels, num_classes)
test_epochs = np.array(test_epochs)
test_labels = create_labels_array(test_labels, num_classes)

# start a computational graph session
tf.reset_default_graph()
sess = tf.Session()

# declare batch size
batch_size = 1

# initialize placeholders and variables
num_features = train_epochs.shape[1]
x_data = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
y_target = tf.placeholder(shape=[num_classes, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[num_classes, batch_size]))

# create Gaussian (RBF) kernel
gamma = tf.constant(-10.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1, 1])
sq_dists = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))


def reshape_matmul(mat, num_classes, _size):
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [num_classes, _size, 1])
    return(tf.matmul(v2, v1))

# compute SVM model
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = reshape_matmul(y_target, num_classes, batch_size)
second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross,
                            y_target_cross)), [1, 2])
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

# create Gaussian (RBF) prediction kernel
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2.0, tf.matmul(x_data,
                      tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
prediction_output = tf.matmul(tf.multiply(y_target, b), pred_kernel)
prediction = tf.argmax(prediction_output - tf.expand_dims(tf.reduce_mean(
                       prediction_output, 1), 1), 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target, 0)),
                          tf.float32))

# declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# training loop
loss_vec = []
for i in range(len(train_epochs)):
    x_input = train_epochs[[i]]
    y_input = train_labels[:, [i]]
    sess.run(train_step, feed_dict={x_data: x_input, y_target: y_input})
    temp_loss = sess.run(loss, feed_dict={x_data: x_input, y_target: y_input})
    loss_vec.append(temp_loss)
    if (i+1) % 25 == 0:
        print('Step #' + str(i+1))
        print('Loss = ' + str(temp_loss))

# prediction loop
batch_accuracy = []
for i in range(len(test_epochs)):
    x_input = test_epochs[[i]]
    y_input = test_labels[:, [i]]
    acc_temp = sess.run(accuracy, feed_dict={x_data: x_input,
                        y_target: y_input, prediction_grid: x_input})
    batch_accuracy.append(acc_temp)

# Plot batch accuracy
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
