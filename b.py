import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version r0.10
from sklearn import metrics
import os
import pywt
import numpy as np
# Useful Constants
# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]
# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
] # Useful Constants


TRAIN = "train/"
TEST = "test/"

DATA_PATH = "Data2/"
DATASET_PATH = DATA_PATH + "UCI Data Dataset/"
# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)






# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

# Input Data

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep


# LSTM Neural Network's internal structure

n_hidden = 32 # Hidden layer num of features
n_classes = 6 # Total classes (should go up, or should go down)


# Training

learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 200
display_iter = 30000  # To show test set accuracy during training


# Some debugging info
print("Some useful info to get an insight on dataset's shape and normalisations:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print((X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test)))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

def LSTM_RNN(_X, _weights, _biases):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
    # Note, some code of this notebook is inspired from an slightly different
    # RNN architecture used on another dataset:
    # https://tensorhub.com/aymericdamien/tensorflow-rnn

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split Data2 because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.nn.rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    #print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    #print(len(_X))
    #print(_X[0].get_shape())
    #print(len(outputs))
    #print(len(outputs[0]))
    #print(_X.get_shape)
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']

def LSTM_RNN2(_X, _weights, _biases):
    #_X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    #_X = tf.reshape(_X, [-1, n_input])
    #_X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    #_X = tf.split(0, n_steps, _X)
    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)


    u_ = tf.Variable(tf.random_normal(shape=[1, n_steps]), name="u_w")
    w_ones = tf.Variable(tf.constant(1.0, shape=[n_steps,1]),name="u_w_one")
    val, state = tf.nn.dynamic_rnn(lstm_cell, _X, dtype=tf.float32)
    val_ = tf.reshape(val,(-1,n_hidden))
    weight_h = tf.Variable(tf.truncated_normal([n_hidden, n_steps]),name='weight_h')
    bias_h = tf.Variable(tf.constant(0.1, shape=[n_steps]))
    u_levels = tf.reshape((tf.matmul(val_, weight_h) + bias_h),(-1,n_steps,n_steps))
    
    #u_levels_ = tf.transpose(u_levels,[0,2,1])
    u_levels_ = tf.transpose(u_levels,[2,0,1])

    u_levels_ = tf.reshape(u_levels_,(n_steps,-1))
    u_levels_t = tf.exp(tf.matmul(u_,u_levels_))
    w_t = tf.reshape(u_levels_t,(-1,n_steps))
    w_ = tf.matmul(w_t,w_ones)
    u_w = tf.div(w_t, w_)
    u_w = tf.reshape(u_w,(-1,1,n_steps))
    m_t = tf.reshape(tf.matmul(u_w,val),(-1,n_hidden))

    prediction = tf.matmul(m_t, _weights['out']) + _biases['out']

def Multi_Scale_Wavelet(trainX,trainY,level,is_multi=True,wave_type='db1'):
    trainX = np.transpose(trainX,[2,0,1])#(7352,128,9) -> (9,7352,128)
    print(trainX.shape)
    temp = [[] for i in range(level)]
    #temp = [[] for i in range(level + 1)]
    N = trainX.shape[-1]
    if (is_multi == True) and (level > 1):
        for i in range(level):
        #for i in range(level+1):
            x1 = []
            for _feature in range(len(trainX)):
                x2 = []
                for sample in range(len(trainX[0])):
                    coeffs = pywt.wavedec(trainX[_feature][sample], wave_type, level=level)
                    current_level = level  - i
                    #current_level = level + 1 - i
                    for j in range(i+1,level+1):
                        coeffs[j] = None
                    _rec = pywt.waverec(coeffs, wave_type)
                    x2.append(_rec[:N])
                #x.append(coeffs[i+1])
                x1.append(x2)

            #temp[current_level - 1].extend(np.transpose(np.array(x1)))
            temp[current_level - 1].extend(np.array(x1))


    else:
        for tab in range(level):
            current_level = level - tab
            temp[current_level - 1].extend(trainX)
    #print("HEHE")
    #print((np.array(temp)).shape)
    final_ = np.transpose(np.array(temp),[2,0,3,1])

    return final_,trainX,np.array(trainY)
def LSTM_RNN3(_X, _weights, _biases):

    data_train = tf.transpose(_X, [0, 2, 1, 3])#samples,levels,time_step,n_input
    data_train = tf.reshape(data_train,(-1,7,n_input))

    with tf.variable_scope('1stlayer_hl'):
        lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        #lstm_cell_bottom = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.tanh,state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        val_bottom, state_bottom = tf.nn.dynamic_rnn(lstm_cell, data_train, dtype=tf.float32)
        temp_b = tf.transpose(val_bottom, [1, 0, 2])
        last_b = tf.gather(tf.transpose(val_bottom, [1, 0, 2]), int(temp_b.get_shape()[0]) - 1)

    temp_ = tf.reshape(last_b,(-1,n_steps,n_hidden))

    with tf.variable_scope('2ndlayer_hl'):
        lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        #lstm_cell_top = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.tanh)
        val_top, state_top = tf.nn.dynamic_rnn(lstm_cell, temp_, dtype=tf.float32)

    #weight = tf.Variable(tf.truncated_normal([n_hidden, int(label.get_shape()[1])]),name='weight')
    #bias = tf.Variable(tf.constant(0.1, shape=[label.get_shape()[1]]))
    
    val_t = tf.transpose(val_top, [1, 0, 2])
    last_t = tf.gather(val_t, int(val_t.get_shape()[0]) - 1)
    #prediction = tf.nn.softmax(tf.matmul(last_t, weight) + bias)






    prediction = tf.matmul(last_t, _weights['out']) + _biases['out']


    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.tanh)
    ##val, state = tf.nn.dynamic_rnn(lstm_cell, _X, dtype=tf.float32)
    #val, state = tf.nn.rnn(lstm_cell, _X, dtype=tf.float32)
    ##val = tf.transpose(val, [1, 0, 2])
    ##last = tf.gather(val, int(val.get_shape()[0]) - 1)    
    ##prediction = tf.nn.softmax(tf.matmul(last, _weights['out']) + _biases['out'])
    ##prediction =  tf.matmul(last, _weights['out']) + _biases['out']
    return prediction




def extract_batch_size(_train, step, batch_size):
    #print("abc")
    #print(len(_train))
    #print(len(_train[0]))
    #print(len(_train[0][0]))
    #print(len(_train[0][0][0]))
    #print(type(_train[0][0]))

    # Function to fetch a "batch_size" amount of Data2 from "(X|y)_train" Data2.

    shape = list(_train.shape)
    shape[0] = batch_size
    #print(shape)
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index]
        #print(_train[index].shape)
        #batch_s.append(_train[index])
    return batch_s


def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


X_train,_,y_train = Multi_Scale_Wavelet(X_train,y_train,level=7)
X_test,_,y_test = Multi_Scale_Wavelet(X_test,y_test,level=7)
print(X_train.shape)
print(y_train.shape)
# Graph input/output
x = tf.placeholder(tf.float32, [None, 7,n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = LSTM_RNN3(x, weights, biases)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the Data2
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) + l2 # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.initialize_all_variables()
sess.run(init)

# Perform Training steps with "batch_size" amount of example Data2 at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs = extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))

    # Fit training using batch Data2
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs,
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)

    # Evaluate network only at some steps for faster training:
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):

        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))

        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy],
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Finished!")

# Accuracy for test Data2

one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))
