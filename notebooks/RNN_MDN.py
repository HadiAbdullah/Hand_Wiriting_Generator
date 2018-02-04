
# coding: utf-8

# #### Implementaiton of the Alex Graves Paper for Hand Writing Synthesis

# ## Imports

# In[1]:

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sys 
import time
import random
import math as m
tf.set_random_seed(1)

sys.path.insert(0,'..')
from utils import plot_stroke
get_ipython().magic('matplotlib inline')
from collections import namedtuple
import datetime


# ## Varaibles

# In[5]:

# MDN and RNN Params
state_size = 300
num_classes = 121
num_layers = 3
features = 3
dropout_pkeep = 0.8
gradient_clip = 10
total_mixtures = 19


# In[7]:

# Model Training Params
batch_size = 3
back_prop_length = 100
seqlength = 500
NEPOCH = 2000
batch_length = batch_size*seqlength*features
save_path = "../models/model1.ckpt"


# In[8]:

# Structs
normal_struct = namedtuple("normalStruct", "x mean sigma")
normal_struct_temp = namedtuple("normalStructTemp", "mean sigma")



def generate_unconditionally(random_seed=1):
    tf.set_random_seed(random_seed)
    
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(sess, save_path)
    
    # Initiate coordinates and State
    _x1 = 0
    _x2 = 0
    _eos = 1
    prev_coordinates = np.array([[[_eos,_x1,_x2]]]) # shape should be [batch,seq_length,features]
    _current_state = np.zeros((num_layers, 2, 1, state_size))


    points = []
    all_coordinates = []
    for i in range(500):
        _eos_rounded= 1

        # Run Session
        feed = {X: prev_coordinates, init_state: _current_state}
        [_eos, _pi, _rho, _normal_struct1, _normal_struct2],_current_state = sess.run([coeffs,current_state], feed_dict = feed)

        # Extract next point
        pi_index = np.argmax(_pi)
        _x1, _x2 = bivarate_normal_sample(random_seed, pi_index,_rho,_normal_struct1, _normal_struct2)
        _eos_rounded = np.round(_eos)

        if(np.isnan(_x1)|np.isnan(_x2)):
#             print("nan found", i)
            continue

        # Replace old point with new point
        prev_coordinates = np.reshape([_eos_rounded,_x1,_x2],np.shape(prev_coordinates))

        # Append new points to the list of all predicted points    
        all_coordinates = np.append(all_coordinates,prev_coordinates)
        all_coordinates = all_coordinates.reshape([-1,3])
    return all_coordinates

# ## Load and Process Stroke Data

# In[9]:

def load_and_process_data(): 
    strokes = np.load('../data/strokes.npy')
    texts = ""
    with open('../data/sentences.txt') as f:
        texts = f.readlines()

    grand_strokes1 = []

    for i in range(0,1000):
        grand_strokes1 = np.append(grand_strokes1,strokes[i],)

    grand_strokes2 = []
    for i in range(1000,2000):
        grand_strokes2 = np.append(grand_strokes2,strokes[i],)

    grand_strokes3 = []
    for i in range(2000,3000):
        grand_strokes3 = np.append(grand_strokes3,strokes[i],)

    grand_strokes4 = []
    for i in range(3000,4000):
        grand_strokes4 = np.append(grand_strokes4,strokes[i],)

    grand_strokes5 = []
    for i in range(4000,5000):
        grand_strokes5 = np.append(grand_strokes5,strokes[i],)

    grand_strokes6 = []
    for i in range(5000,6000):
        grand_strokes6 = np.append(grand_strokes6,strokes[i],)


    g1 = np.append(grand_strokes1,grand_strokes2)
    g2 = np.append(grand_strokes3,grand_strokes4)
    g3 = np.append(grand_strokes5,grand_strokes6)

    g12 = np.append(g1,g2)
    grand_strokes = np.append(g12,g3)
    grand_strokes = grand_strokes.reshape([-1,3])
    print("Data Ready")
    return grand_strokes


# ## Mixed Density Network

# #### MDN Coefficients

# In[10]:

# Equation 18
def get_stroke(eos_predicted):
    eos_predicted =  tf.reciprocal(1+tf.exp(eos_predicted))  
    return eos_predicted

# Equation 19    
def get_weights(weights):
    weights = tf.nn.softmax(weights)
    return weights

# Equation 20    
def get_mean(means):
    return means

# Equation 21    
def get_stdv(standard_deviations):
    out_sigma = tf.exp(standard_deviations)
    return out_sigma

# Equation 22    
def get_correlation(correlations):
    corr = tf.tanh(correlations)
    return corr


# #### Gaussian Loss

# In[11]:

def normal1d(normal_struct):
    x_mu = tf.subtract(normal_struct.x, normal_struct.mean)
    return tf.square(tf.divide(x_mu, normal_struct.sigma))

# Equation 25
def Z(normal_struct1,normal_struct2, correlation):
    x11 = normal1d(normal_struct1)
    x12 = normal1d(normal_struct2)
    x1x2 = tf.divide((tf.multiply(
                      np.subtract(normal_struct1.x,normal_struct1.mean),\
                      tf.subtract(normal_struct2.x,normal_struct2.mean))),\
                     (tf.multiply(normal_struct1.sigma,normal_struct2.sigma)))

    Z = x11+x12-(2*tf.multiply(correlation,x1x2))

    return Z

# Equation 24
def normal2d(normal_struct1,normal_struct2, correlation):
    pi_constant = tf.constant(m.pi)
    
    z = Z(normal_struct1,normal_struct2, correlation)
    correlation_sqr = 1-(correlation*correlation)
    expo = tf.exp(tf.div(-z,(2*(correlation_sqr))))

    sigma = tf.multiply(normal_struct1.sigma,normal_struct2.sigma)
    denominator = tf.reciprocal(2*pi_constant*tf.multiply(sigma, tf.sqrt(correlation_sqr)))
    
    result = tf.multiply(expo,denominator)
    
    return result



def loss_normal(gaussian, weights):
    normal_loss1 = tf.multiply(gaussian, weights)
    normal_loss2 = tf.reduce_sum(normal_loss1,1,keep_dims=True) 
    normal_loss2 = tf.maximum(normal_loss2, 1e-20) # This is done or loss will be NaN or Inf
    normal_loss3 = -tf.log(normal_loss2) 
    
    return normal_loss3

def loss_eos(eos_actual,eos_predicted):
    i = tf.constant(0)
    values_eos =  tf.Variable([],dtype=tf.float32)
    
    # Applying correct function to EOS as per Equation 26
    def condition(i,values_eos):
        return i<(batch_size*back_prop_length)
    
    def body(i,values_eos):
        result = tf.cond(eos_actual[i][0] < 1, lambda:-tf.log(1-eos_predicted[i]), lambda: -tf.log(eos_predicted[i]))
        values_eos = tf.concat([values_eos,result],0)
        return tf.add(i, 1),values_eos

    # do the loop:
    r,eos_result_log  = tf.while_loop(condition, body, [i,values_eos],shape_invariants=[i.get_shape(),
                                                   tf.TensorShape([None])])
    return eos_result_log

# Equation 26
def loss(weights,normal_struct1,normal_struct2, correlation, eos_actual, eos_predicted):

    gaussian = normal2d(normal_struct1,normal_struct2, correlation)
    normal = loss_normal(gaussian, weights)
    
    eos = loss_eos(eos_actual,eos_predicted)
    eos = tf.reshape(eos,tf.shape(normal))
    
    result = tf.reduce_sum(normal + eos)
    
    return result,normal,eos,gaussian


# #### MDN Distribution

# In[12]:

def mdn_coeffs(mixture):

        
    eos_predicted = mixture[:, 0:1]
    mixture = mixture[:, 1:]

    #Split remaining values equally among the mix coeffs     
    "Section 4.2:(20 weights, 40 means, 40 standard deviations and 20 correlations)"
    weights, mean1, mean2, standard_deviation1, standard_deviation2, correlations = tf.split(mixture, 6, 1)              

    # Apply functions to the extracted values
    eos_predicted = get_stroke(eos_predicted)
    weights = get_weights(weights)
    mean1 = get_mean(mean1)
    mean2 = get_mean(mean2)
    sigma1 = get_stdv(standard_deviation1)
    sigma2 = get_stdv(standard_deviation2)
    correlation = get_correlation(correlations)
    

    normal_struct1 = normal_struct_temp( mean=mean1, sigma=sigma1)
    normal_struct2 = normal_struct_temp( mean=mean2, sigma=sigma2)

    return [eos_predicted, weights, correlation, normal_struct1, normal_struct2]

    
# Sample from the a bivariate sample of the given parameters
def bivarate_normal_sample(random_seed,pi_index,_rho,_normal_struct1, _normal_struct2):
    random.seed(random_seed)
    mean = [_normal_struct1.mean[0][pi_index], _normal_struct2.mean[0][pi_index]]

    # Covariance Matrix: http://mathworld.wolfram.com/BivariateNormalDistribution.html
    # |    (sigma1)^2       rho*sigma1*sigma2 |
    # | rho*sigma1*sigma2       (sigma1)^2    | 
    covariance_matrix = [[np.square(_normal_struct1.sigma[0][pi_index]),                       (_rho[0][pi_index]*_normal_struct1.sigma[0][pi_index]*_normal_struct2.sigma[0][pi_index])],                      [(_rho[0][pi_index]*_normal_struct1.sigma[0][pi_index]*_normal_struct2.sigma[0][pi_index]),                       np.square(_normal_struct2.sigma[0][pi_index])]]

    point = np.random.multivariate_normal(mean, covariance_matrix, 1).T

    _x1 = point[0][0]
    _x2 = point[1][0]
    
    return _x1,_x2




# ## RNN Variables

# In[13]:

# X: The batch input coordinates [EOS,x1,x2]
X = tf.placeholder(tf.float32, [None, None,features]) # [ batch_size, SEQLEN,features ]

# Y: The predictions 
Y = tf.placeholder(tf.float32, [None,features]) # [ batch_size, SEQLEN,features ]

# State of the RNN
init_state = tf.placeholder(tf.float32, [num_layers, 2, None, state_size])

# Weight and Bias tensors
W = tf.Variable(np.random.rand(state_size, total_mixtures),dtype=tf.float32)
bias = tf.Variable(np.zeros((1,total_mixtures)), dtype=tf.float32)


# ## RNN Model

# In[14]:

# Keeping track of intermediate state of the RNNs
state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )

# RNNs stacked ontop of each other
'Section 4.2: three hidden layers, each consisting of 400 LSTM cells'
stacked_rnn = []
for _ in range(num_layers):
    '4.2: network was retrained with adaptive weight noise ... with all std. devs. initialised to 0.075'
    adaptive_weight_noise = tf.truncated_normal_initializer(stddev=0.075)
    cell = tf.nn.rnn_cell.LSTMCell(state_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_pkeep)
    stacked_rnn.append(cell)

MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn , state_is_tuple=True)
states_series,current_state = tf.nn.dynamic_rnn(cell= MultiRNNCell,inputs= X, initial_state= rnn_tuple_state)
# states_series: [ batch_size, SEQLEN, state_size ]
# current_state  [ batch_size, state_size*num_layers ]

states_series = tf.reshape(states_series, [-1, state_size])
mixtures = tf.matmul(states_series, W) + bias 
# Split the Y 
[eos_actual,x1,x2] = tf.split(Y, features, 1) 


# ## Loss and Optimization

# In[15]:

# Get Mixture coeffs
coeffs = mdn_coeffs(mixtures)
[eos_predicted, weights, correlation, normal_struct1, normal_struct2] = coeffs

# Create a struct of mixture coeffs and coordinate values
normal_struct1 = normal_struct(x=x1, mean=normal_struct1.mean, sigma=normal_struct1.sigma)
normal_struct2 = normal_struct(x=x2, mean=normal_struct2.mean, sigma=normal_struct2.sigma)

# Calcuate Loss
calcualted_loss,_,_,_ = loss(weights,normal_struct1,normal_struct2, correlation, eos_actual, eos_predicted)

# Normalize loss over batch
# https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
calcualted_loss = tf.divide(calcualted_loss,batch_size)

# Apply Gradient Clipping:
optimizer = tf.train.AdamOptimizer(learning_rate=0.002)
'Section 4.2: LSTM derivates were clipped in the range [−10, 10]'
gradients, variables = zip(*optimizer.compute_gradients(calcualted_loss))
gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)
optimize = optimizer.apply_gradients(zip(gradients, variables))



# ## Train Model

# In[16]:

def train(data):
    # Batches
    start_batch = 0
    end_batch = batch_length
    batches = 10000000
    _total_loss = []
    epochs = 0
    saver = tf.train.Saver()
    
    grand_strokes = data
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    # Add ops to save and restore all the variables.

    _current_state = np.zeros((num_layers, 2, batch_size, state_size))

    index = 0
    while(True):
        index += 1
        print(index)
        print(datetime.datetime.now())

        if(start_batch>grand_strokes.shape[0]):
            epochs +=1
            start_batch = start_batch%grand_strokes.shape[0]
            end_batch = end_batch%grand_strokes.shape[0]
            print("epochs ",epochs)


        single_batch = grand_strokes.take(range(start_batch,end_batch),mode = "wrap")
        single_batch = np.reshape(single_batch,[batch_size,seqlength,features])

        start_batch += batch_size
        end_batch += batch_size

        # Mini Batches
        mini_lenght = back_prop_length
        start_mini = 0 
        end_mini = mini_lenght

        if(epochs>NEPOCH):
            print("Complete")
            break

        while(True):
            if(end_mini+1>single_batch.shape[1]):
                break

            batchX = single_batch[:,start_mini:end_mini]
            batchY = single_batch[:,start_mini+1:end_mini+1]
            batchY = batchY.reshape([-1,3])
            start_mini += 1
            end_mini += 1
            _,_loss,  __current_state =            sess.run([optimize,calcualted_loss, current_state],
                 feed_dict={X: batchX, 
                            Y: batchY,
                            init_state: _current_state })


            _total_loss = np.append(_total_loss,_loss) 

            # Quit if a NaN is found in the Loss values
            if(np.isnan(_loss)):
                print("Naan found")
                continue
            else:
                _current_state = __current_state
            

        # Create a checkpoint in every iteration
        saver.save(sess, save_path)

