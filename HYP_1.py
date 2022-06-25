# Multiple Inputs
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras import backend as k
import tensorflow as tf
from keras.layers.core import Lambda

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
#from keras.utils import plot_model

from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout, Dot
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import GlobalAveragePooling1D, Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Embedding, LSTM, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
import math
#import flipGradientTF 
from keras.optimizers import Adam

from KLDIV import klDiv

np.random.seed(0)

opt = Adam(learning_rate=0.0001)#0.0001

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad
    
def gradientRev(x):
	return grad_reverse(x) 

def checkModel(model):
	names = [weight.name for layer in model.layers for weight in layer.weights]#; print names
	weights = model.get_weights()
	d={str(names[i]).split("/")[0]:k.variable(weights[i]) for i in range(len(names)) if i%2==0}
	return d

class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def constraint(y_true, y_pred, D_Name, S_Name, model):
	names = [weight.name for layer in model.layers for weight in layer.weights]#; print names
	weights = model.get_weights()
	d={str(names[i]).split("/")[0]:k.variable(weights[i]) for i in range(len(names)) if i%2==0}#; print d.keys()
	#print "**********\n",d[D_Name].shape, "******\n", d[S_Name].shape, "\n"
	kdot=tf.matmul(d[D_Name], k.transpose(d[S_Name]))
	return scalar+k.sum(k.abs(y_true-y_pred))

def orthoLoss(D_Name, S_Name, model):
	def stdLoss(y_true, y_pred):
		return constraint(y_true, y_pred, D_Name, S_Name, model)
	return stdLoss

def loss1(y_true, y_prediction):
	ll= y_true - keras.softmax(U*x1_output, V*s_output)

def KL_divergence(x1, y1):
    return x1* tf.log(x1 / y1) + (1 - x1) * tf.log((1 - x1) / (1 - y1))

def kl_divergence1(y_true, y_pred):
    return y_true * np.log(y_true / y_pred)


def custom_loss(y_true, y_pred):
	loss= (y_true*y_pred)+((1-y_true)/(y_pred))
	return loss

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def lT1(y_True, y_Pred, x1_Pred, x2_Pred, U, V):
	loss=abs(y_True-sigmoid(np.concatenate(np.matmul(U, x1_pred), np.matmul(V, x2_Pred), axis=0).sum()))
	return loss

def lossTask1(x1_Pred, x2_Pred, U, V):
	def lossRet(y_True, y_pred):
		return lT1(y_True, y_Pred, x1_Pred, x2_Pred, U, V)
	return lossRet

def myFun(x, M):
	x=k.transpose(tf.matmul(M, k.transpose(x)))
	return x

def repreNN(**params):
	##parameter 
	MAX_SEQUENCE_LENGTH=params["MAX_SEQUENCE_LENGTH"]
	
	#embedding layer
	input_dim=params["vocab_size"]
	output_dim=params["output_word_dim"]
	input_sequence_length=params["input_sequence_length"]

	#con1D
	no_of_filters=params["no_of_filters"]
	filter_size=params["filter_size"]
	
	#MaxPool
	maxPool_size=params["maxPool_size"]
	
	#Dropout
	dropout_fraction=params["dropout_fraction"]

	#LSTM
	lstm_output_dim=params["lstm_output_dim"]

	#Dense
	dense_dim_1=params["dense_dim_1"]
	dense_dim_2=params["dense_dim_2"]

	main_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	x = Embedding(output_dim=output_dim, input_dim=input_dim, input_length=input_sequence_length)(main_input)
	x = Conv1D(no_of_filters, filter_size)(x)
	# MaxPool divides the length of the sequence by 5
	x = MaxPooling1D(maxPool_size)(x)
	x = Dropout(dropout_fraction)(x)
	x = LSTM(lstm_output_dim, activation="sigmoid", dropout=0.5, recurrent_dropout=0.3)(x)
	x=Dense(dense_dim_1, activation="sigmoid")(x)
	#x=Dense(dense_dim_1, activation="sigmoid")(x)
	#x=Dense(dense_dim_1, activation="sigmoid", kernel_regularizer='l1')(x)
	x_i=Dense(dense_dim_1, activation="sigmoid", name="denseVec_Hyp")(x)
	
	x_output= Dense(1, activation= 'sigmoid', name="tarPred")(x_i)
	#x_output = Dense(1, activation='sigmoid')(x)
	
	#dp_output= Dense(32, activation= 'sigmoid')(x_i) ## dp_output=x_i
	
	return main_input, x, x_i, x_output#dp_output #input layer, output tensor, final output tensor

def hardMTL(**params):
	##parameter 
	N=params["N"]	
	j=params["j"]
	MAX_SEQUENCE_LENGTH=params["MAX_SEQUENCE_LENGTH"]
	
	#embedding layer
	input_dim=params["vocab_size"]
	output_dim=params["output_word_dim"]
	input_sequence_length=params["input_sequence_length"]

	#con1D
	no_of_filters=params["no_of_filters"]
	filter_size=params["filter_size"]
	
	#MaxPool
	maxPool_size=params["maxPool_size"]
	
	#Dropout
	dropout_fraction=params["dropout_fraction"]

	#LSTM
	lstm_output_dim=params["lstm_output_dim"]

	#Dense
	dense_dim_1=params["dense_dim_1"]
	dense_dim_2=params["dense_dim_2"]

	visL=[]
	for i in range(N):
		visL.append(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
	s=concatenate(visL)
	s = Embedding(output_dim=output_dim, input_dim=input_dim, input_length=N*input_sequence_length)(s)

	s = Conv1D(no_of_filters, filter_size)(s)
	# MaxPool divides the length of the sequence by 5
	s = MaxPooling1D(maxPool_size)(s)
	s = Dropout(dropout_fraction)(s)
	s = LSTM(lstm_output_dim, activation="sigmoid", dropout=0.5, recurrent_dropout=0.3)(s)#kernel_regularizer="l1"
	s=Dense(dense_dim_1, activation="sigmoid")(s)
	#s=Dense(dense_dim_1, activation="sigmoid", kernel_regularizer="l1")(s)
	s1=Dense(dense_dim_1, activation="sigmoid")(s)
	#s=Dense(dense_dim_1, activation="sigmoid")(s)
	#s_output = Dense(N, activation='sigmoid')(s1)
	d_output1= Dense(1, activation= 'sigmoid')(s1)
	d_output2= Dense(1, activation= 'sigmoid', name="tarPredDuo_"+str(j))(s1)
	return visL, s1, d_output1, d_output2#visL, s, s_output, d_output 
	
def advOrthoMTL(**kwargs):
	
	N=kwargs["N"]
	#Dense
	dense_dim_1=kwargs["dense_dim_1"]
	dense_dim_2=kwargs["dense_dim_2"]
	
	#vis_mtlL, mtl_output, d_output1, d_output2=hardMTL(**kwargs)
	
	vis_srcL=[]; vis_tarL=[]; 
	pre_srctarL=[]
	output_srcL=[]; output_tarL=[]
	kwargs["N"]=2
	for i in range(N-1):
		kwargs["j"]=i#; print ("*******************", i); input("enter")
		vis_mtlL, pre_mtl, d_output1, d_output2=hardMTL(**kwargs)
		#print (len(vis_mtlL)); input("enter")
		src, tar=vis_mtlL
		vis_srcL.append(src); vis_tarL.append(tar)
		pre_srctarL.append(pre_mtl)
		output_srcL.append(d_output1); output_tarL.append(d_output2)
	
	vis, oS, dp_output, oS_output=repreNN(**kwargs)

	#concatenation of shared models
	s=pre_srctarL
	if len(pre_srctarL)==1:
		s=pre_srctarL[0]
	else:
		s=concatenate(pre_srctarL)#; print (s.shape, dense_dim_1, oS.shape); input("enter")
	s=Dense(dense_dim_1, activation= 'sigmoid')(s)	
	#s=concatenate(pre_srctarL)#; print ("shape of both ", s.shape, oS.shape); input("enter")
	new_output= concatenate([oS, s])
	new_output= Dense(dense_dim_1, activation= 'sigmoid')(new_output)
	new_output=klDiv(5, name="RAD")(new_output)
	new_output= Dense(1, activation= 'sigmoid', name="tarCustomPred")(new_output)
	
	model1=Model(vis_srcL+vis_tarL+[vis], output_srcL+output_tarL+[oS_output, new_output])# N+1 tar first input, N+2 tar first output
	lossL=["mean_squared_error"]+["mean_squared_error"]+["mean_squared_error" for i in range(len(output_tarL)+len(output_srcL))]
	model1.compile(optimizer=opt,loss=lossL, metrics=["accuracy"])
		
	#model summary
	#print(model1.summary())
	#plot graph
	#plot_model(model1, to_file='ensem_Draw.png')

	return model1

params={"N":3, "MAX_SEQUENCE_LENGTH":80, "vocab_size":10000, "output_word_dim":200, "input_sequence_length":150, "no_of_filters":64, "filter_size":5, "maxPool_size":5, "dropout_fraction":0.5, "lstm_output_dim":64, "dense_dim_1":64, "dense_dim_2":40}
#advOrthoMTL(**params)
	

