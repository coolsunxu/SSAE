
import torch
import torch.nn as nn

# 编码器
class Encoder(nn.Module):
	
	def __init__(self,fin,fout):
		super(Encoder,self).__init__()
		
		self.fin = fin
		self.fout = fout

		self.fc = nn.Linear(self.fin,self.fout)

	def forward(self,x):
		return torch.relu(self.fc(x))


# 解码器
class Decoder(nn.Module):
	
	def __init__(self,fin,fout):
		super(Decoder,self).__init__()

		self.fin = fin
		self.fout = fout

		self.fc = nn.Linear(self.fin,self.fout)
		self.resweight = nn.Parameter(torch.Tensor([0])) # 线性函数
		self.bias = nn.Parameter(torch.Tensor(self.fout))

	def forward(self,x):
		x = self.fc(x)
		return self.resweight*x+self.bias

# 多层编码器
class ENBLOCK(nn.Module):
	def __init__(self, units):
		super(ENBLOCK, self).__init__()
		self.units = units
		self.n_layer = len(self.units) - 1 # layers size
		self.layer_stack = nn.ModuleList() # module list

		for i in range(self.n_layer):
			self.layer_stack.append( 
				Encoder(self.units[i], self.units[i+1])
			)

	def forward(self, x, F):
		
		for _, encoder_layer in enumerate(self.layer_stack):
			x = encoder_layer(x)
			F.append(x)
			
		return x

# 多层解码器
class DEBLOCK(nn.Module):
	def __init__(self, units):
		super(DEBLOCK, self).__init__()
		self.units = units
		self.n_layer = len(self.units) - 1 # layers size
		self.layer_stack = nn.ModuleList() # module list

		for i in range(self.n_layer):
			self.layer_stack.append( 
				Decoder(self.units[i], self.units[i+1])
			)

	def forward(self, F, G):
		
		for i, decoder_layer in enumerate(self.layer_stack):
			x = decoder_layer(F[self.n_layer-i-1])
			G.append(x)
			
		return x

class SSAE(nn.Module):
	
	def __init__(self,units):
		super(SSAE,self).__init__()
		
		self.units = units
		self.enblock = ENBLOCK(self.units)
		units.reverse(); # 反转
		self.deblock = DEBLOCK(self.units)
		
	def forward(self,x):
		F = []
		G = []
		d = self.enblock(x,F) # 潜在空间变量采样
		y = self.deblock(F,G) # 重构的输出

		return F,G



units = [8,256,128,64]
a = torch.randn(5,8)

s = SSAE(units)
F,G = s(a)
print(len(F),len(G))
print(F[2].shape,G[2].shape)
