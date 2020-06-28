import time
import numpy as np
import h5py
# from matplotlib import pyplot as plt
import scipy
# from PIL import Image
from scipy import ndimage
# from dnn_app_utils_v3 import *


# plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
from regular_deep import sigmoid, relu,relu_backward, sigmoid_backward, initialize_parameters_deep
from regular_deep import linear_forward, linear_activation_forward, L_model_forward
from regular_deep import linear_backward,linear_activation_backward, update_parameters, predict
from regular_deep import L_model_backward
from regular_deep import compute_cost, L_model_backward
# from L_layer_model import L_layer_model
np.random.seed(1)


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from pandas import read_csv
import pandas as pd

df = read_csv('Half_EN1_perf_n_build_merged_n_missing_filled_w_only_inputs_w_tags.csv',index_col=False)
import statistics as st
df['SN_BM']=df['SN_BM'].fillna(0)
df['SN_TRANS_DUCT']=df['SN_TRANS_DUCT'].fillna(0)
df['SN_EX_NOZ_5']=df['SN_EX_NOZ_5'].fillna(0)
df['SN_SLAVE_EEC']=df['SN_SLAVE_EEC'].fillna(0)
df['ENGINE_MOUNT']=df['ENGINE_MOUNT'].fillna(0)

df['OIL_CONS']=df.fillna(np.mean(df['OIL_CONS']))
df['OIL_CONS']=df.fillna(st.mode(df['SN_STRBK']))
features_with_no_missing_values = []
for i in df.columns:
#     print('the number of missing values in column ', i, 'is ', df[i].isna().sum())
    if df[i].isna().sum() == 0:
        features_with_no_missing_values.append(i)
# print('features_with_no_missing_values = ',features_with_no_missing_values)
# print('The number of features_with_no_missing_values = ',len(features_with_no_missing_values))
#df.isna().sum
for i in df.columns:
    if i not in features_with_no_missing_values:
        df = df.drop(columns = i)
df = df.drop(columns =['SN_SLAVE_EEC','ENGINE_MOUNT','SN_EX_NOZ_5','SN_TRANS_DUCT','SN_BM','SN','ROW_C_DATE','SFC','CAL_DATE','TESTCELL','ACCEPT_SUF','OIL_CONS'])
print('df = ')
print(len(df.columns))
target = df['0.5EN1 Present']
predictors = df.drop(columns=['0.5EN1 Present'])

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
target = lb_make.fit_transform(target)
# classes_1D[["0", "1"]].head(11)
target =pd.DataFrame(target )
assert(target.shape== (target.shape[0],1))


X_train, X_test, y_train, y_test=train_test_split(predictors, target, test_size=0.2, random_state=42)


print ("train_x's shape: " + str(X_train.shape))
print ("test_x's shape: " + str(X_test.shape))
print ("train_y's shape: " + str(y_train.shape))
# Number of training examples: m
# Number of testing examples: n
# Each image is of size: (64, 64, 3)
# train_x_orig shape: (209, 64, 64, 3)
# train_y shape: (1, m)
# test_x_orig shape: (50, 64, 64, 3)
# test_y shape: (1, n)
#train_x's shape: (12288, m)
# test_x's shape: (12288, n)
# n_x = 12288     # num_px * num_px * 3
# n_h = 7
# n_y = 1
# layers_dims = (n_x, n_h, n_y)
    
X_train=np.asarray(X_train).T
X_test=np.asarray(X_test).T


y_train=np.asarray(y_train.T)
y_test=np.asarray(y_test.T)

print ("train_x's shape: " + str(X_train.shape))
print ("test_x's shape: " + str(X_test.shape))
print ("train_y's shape: " + str(y_train.shape))

# n_x = 12288     # num_px * num_px * 3
# n_h = 7
# n_y = 1
# layers_dims = (n_x, n_h, n_y)
n_x = X_train.shape[0]    # num_px * num_px * 3
print(X_train.shape[1])
print(n_x)


n_h = 7
n_y = 9
layers_dims = [n_x , 20, 5, 1] #  4-layer model


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, lambd = 0):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        # print('caches:',caches)
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y, parameters, lambd)
        ### END CODE HERE ###
        # print('parameters 1', parameters)
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches, lambd)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            cost = compute_cost(AL, Y, parameters, lambd)
            print ("Cost after iteration %i: %f" %(i, cost))
            
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    
    return parameters


parameters = L_layer_model(X_train, y_train, layers_dims, num_iterations = 2000, print_cost = True, lambd=0.7)
pred_train = predict(X_test, y_test, parameters)