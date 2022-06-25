import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
#random.seed(0)
import numpy as np
#np.random.seed(0)
import tensorflow as tf
#tf.random.set_seed(1)
#tf.compat.v1.get_default_graph()
#from tensorflow import set_random_seed
#set_random_seed(1)
import sys
import pandas as pd
#import numpy as np
#np.random.seed(0)
import keras
from keras.layers import Layer
import keras.backend as K
#import tensorflow as tf
from keras.utils import plot_model
from keras import Input, Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Flatten, Dense, Lambda, LSTM, Reshape, TimeDistributed, Dropout, BatchNormalization, Concatenate, Reshape, Embedding, Conv1D, MaxPooling1D
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Bidirectional

#from KER import Ker

opt = Adam(lr=0.0001)
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)

def l1_reg(weight_matrix):
    return 0.00001 * K.sum(K.abs(weight_matrix))
    
@tf.custom_gradient
def grad_reverse(x):
	y = tf.identity(x)
	def custom_grad(dy):
		return -dy
	return y, custom_grad		
    
class Pos:

	def __init__(self):
	
		print ("initialized")
		self.N=1
		self.batch_input_shape=(32, 80)
		self.input_shape=(80,)
		self.output_dim=(200)
		self.input_length=(80)
		self.vocab_size=(20000)
		self.no_of_filters=5
		self.filter_size=5
		self.maxPool_size=5
		self.lstm_dim=150
		self.dense_dim=150
		self.count=0
		
	def _proc(self):
	
		print ("protected scope")	
		
	def posModel(self):
		
		#vis=Input(batch_shape=self.batch_input_shape)
		vis=Input(shape=self.input_shape)
		s=Embedding(output_dim=self.output_dim, input_dim=self.vocab_size, input_length=self.input_length)(vis)
		s=Conv1D(self.no_of_filters, self.filter_size)(s)
		s=MaxPooling1D(self.maxPool_size)(s)
		s=LSTM(self.lstm_dim, activation="sigmoid", dropout=0.5, recurrent_dropout=0.3)(s)
		
		s_pos=Dense(self.dense_dim, activation="sigmoid")(s)
		s_pos_output=Dense(self.input_length, activation="sigmoid")(s_pos)
		
		s=Dense(self.dense_dim, activation="sigmoid")(s)
		s=Lambda(lambda x:tf.multiply(x[0], x[1]))([s, s_pos])
		
		s=Dense(self.dense_dim, activation="sigmoid", name="denseVec_"+str(self.count))(s)		
		s_output=Dense(1, activation='sigmoid', name="predLayer_"+str(self.count))(s)
		
		model=Model([vis], [s_output, s_pos_output])
		
		#print model summary
		#print (model.summary())
		#model graph
		#plot_model(model, to_file="repo/pos_model.png")

		#input("once posmodel ")
		
		return (vis, s, s_output, s_pos_output, model)
		
	def posModel4Grp(self, srcL):
		
		eachModel=[]
		for src in srcL:
			eachModel.append(self.posModel())
			self.count=self.count+1
		eachModel.append(self.posModel())##for target
		
		#create a networked model for pos using all src and target
		visL=[]; outputL=[]; pos_outputL=[]
		posVectorL=[]
		for md in eachModel:
			visL.append(md[0])
			outputL.append(md[2])
			pos_outputL.append(md[3])
			posVectorL.append(md[1])
			
		pv=Concatenate()(posVectorL)
		pv=Dense((len(srcL)+1)*self.dense_dim, activation="sigmoid")(pv)
		pv=Dense((len(srcL)+1)*self.dense_dim, activation="sigmoid")(pv)
		pv=Dense(self.dense_dim, activation="sigmoid")(pv)
		pv=Dense(self.dense_dim, activation="sigmoid")(pv)
		pv=Lambda(lambda x:grad_reverse(x), name="GradRevLayer")(pv)
		pv_output=Dense(len(srcL)+1, activation="sigmoid", name="predLayer_mtl")(pv)
		
		print ("visL ", [ x.shape for x in visL])
		print ("outputL ", [ x.shape for x in outputL])
		print ("pv_output ", pv_output.shape)
		print ("pos_outputL ", [x.shape for x in pos_outputL])		
		
		##model visL=[src...(?, 80), (?, 80)...(?, 80)_target]    input 
		##model outputL=[src....(?, 1), (?, 1)...(?, 1)_target]    classification output
		##model pv_output= (?, src...target len(srcL)+1)        MTL output 
		##model pos_outputL=[src....(?, 80), (?, 80), (?, 80), ....(?, 80)_target]   pos-tag output
		
		outputTerminals=outputL+[pv_output]+pos_outputL
		posFinal=Model(visL, outputTerminals)
		lossL=[ "mse" for lis in outputTerminals ]			
		
		posFinal.compile(optimizer=opt, loss=lossL, metrics=["accuracy"])
		
		#model summary
		print (posFinal.summary())
		#model graph plot
		#plot_model(posFinal, to_file="repo/pos_model.png")

		return (visL, outputL, pv_output, pos_outputL, posFinal)	
		
		
def prog():
	
	srcL, target=[1, 2, 3], 1

	ps=Pos()
	tup=ps.posModel4Grp(srcL)
	return
		
#prog()



