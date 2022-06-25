import keras
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
#import tensorflow_probability as tfp
from keras import activations, initializers, constraints

class klDiv(Layer):

	def __init__(self, out_N=None, kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None, use_bias=True, bias_initializer='zeros', bias_regularizer=None, bias_constraint=None,**kwargs):
		self.out_N=out_N
		self.kernel_initializer = keras.initializers.get(kernel_initializer)
		self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
		self.kernel_constraint = keras.constraints.get(kernel_constraint)
		self.use_bias = use_bias
		self.bias_initializer = keras.initializers.get(bias_initializer)
		self.bias_regularizer = keras.regularizers.get(bias_regularizer)
		self.bias_constraint = keras.constraints.get(bias_constraint)
		self.kernel, self.b = None, None
		super(klDiv, self).__init__(**kwargs)

	def build(self, input_shape):
		#assert isinstance(input_shape, list)
		bt, d=input_shape
		# Create a trainable weight variable for this layer.
		self.hyp = self.add_weight(name='kernel', shape=(d, ), initializer='uniform', trainable=True, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
		
		if self.use_bias:
			self.b = self.add_weight(shape=(d, ), initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint, name='{}_b'.format(self.name),)
		super(klDiv, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		#assert isinstance(x, list)
		bt, d=x.get_shape().as_list()
		#out=tf.multiply(x, self.hyp)
		#out=tf.abs(x-self.hyp)
		#out=tf.abs(x-self.hyp)#; print (" shape of square ", out.shape, x.shape, self.hyp.shape); input("enter")		
		out=tf.sqrt(tf.reduce_sum(tf.square(tf.abs(x-self.hyp)), axis=1, keepdims=True))
		
		if self.use_bias:
			out+= self.b
		out=tf.nn.sigmoid(out)
		#print ("out final ", out.shape); input("enter")
		#out=tf.nn.relu(out)
		return out

	def get_config(self):
		config = super(klDiv, self).get_config()
		config.update({"out_N": self.out_N})
		return config

	def compute_output_shape(self, input_shape):
		#assert isinstance(input_shape, list)
		bt, d=input_shape
		return (bt, d)#[(shape_a[:-1], self.output_dim)]
		###this is important return only exact output shape this custom layer is sending including dim for batch as None or ? 
