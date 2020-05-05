
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,activations,initializers

# 编码器
class Encoder(keras.Model):
	
	def __init__(self,fout):
		super(Encoder,self).__init__()

		self.fout = fout
		self.fc = layers.Dense(self.fout)

	def call(self,x):
		return activations.relu(self.fc(x))


# 解码器
class Decoder(keras.Model):
	
    def __init__(self,fout):
        super(Decoder,self).__init__()

        self.fout = fout

        self.fc = layers.Dense(self.fout)
        self.resweight = tf.Variable(0.0,trainable=True) # 线性函数
        self.initializer = initializers.GlorotUniform()
        self.bias = tf.Variable(self.initializer(shape=[self.fout], dtype=tf.float32))

    def call(self,x):
        x = self.fc(x)
        return self.resweight*x+self.bias

# 多层编码器
class ENBLOCK(keras.Model):
	def __init__(self, units):
		super(ENBLOCK, self).__init__()
		self.units = units
		self.n_layer = len(self.units) - 1 # layers size
		self.layer_stack = [] # module list

		for i in range(self.n_layer):
			self.layer_stack.append( 
				Encoder(self.units[i+1])
			)

	def call(self, x, F):
		
		for _, encoder_layer in enumerate(self.layer_stack):
			x = encoder_layer(x)
			F.append(x)
			
		return x

# 多层解码器
class DEBLOCK(keras.Model):
	def __init__(self, units):
		super(DEBLOCK, self).__init__()
		self.units = units
		self.n_layer = len(self.units) - 1 # layers size
		self.layer_stack = [] # module list

		for i in range(self.n_layer):
			self.layer_stack.append( 
				Decoder(self.units[i+1])
			)

	def call(self, F, G):
		
		for i, decoder_layer in enumerate(self.layer_stack):
			x = decoder_layer(F[self.n_layer-i-1])
			G.append(x)
			
		return x

class SSAE(keras.Model):
	
	def __init__(self,units):
		super(SSAE,self).__init__()
		
		self.units = units
		self.enblock = ENBLOCK(self.units)
		units.reverse(); # 反转
		self.deblock = DEBLOCK(self.units)
		
	def call(self,x):
		F = []
		G = []
		d = self.enblock(x,F) # 潜在空间变量采样
		y = self.deblock(F,G) # 重构的输出

		return F,G



units = [8,256,128,64]
a = tf.random.normal([5,8])

s = SSAE(units)
F,G = s(a)
print(len(F),len(G))
print(F[2].shape,G[2].shape)
