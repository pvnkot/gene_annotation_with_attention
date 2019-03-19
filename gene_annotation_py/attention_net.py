import torch
from torch import nn
import torch
from torch import nn
import utils
import torch.nn.functional as F
import torch.autograd as autograd
import Config

class Attention_Net(nn.Module):
    def __init__(self, kmer_size):
        super(Attention_Net, self).__init__()
        #embeds are of the dimension n * (1 x (embedding_size * no_of_kmers))
        self.embeds = nn.Embedding(len(utils.create_vocabulary(Config.window_size)), Config.embedding_size)
        #Embeds[125, 5]
        self.embeds_size = Config.embedding_size*(kmer_size)
        # embeds_size (495)
        #self.attn_weights = nn.Parameter(autograd.Variable(torch.randn(self.embeds_size, self.embeds_size)))
        self.attn_weights = autograd.Variable(torch.randn(self.embeds_size, self.embeds_size)) #shape[495, 495]
        #attn_weights = autograd.Variable(torch.randn(self.embeds_size, self.embeds_size))
        self.tanh = torch.tanh
        self.fc1 = nn.Linear(self.embeds_size, Config.hidden_layer_size)
        self.relu = F.relu
        #self.context = None
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(Config.hidden_layer_size, 1)
        #self.out = nn.Linear(hidden_layer_size_2, 1)

    def forward(self, inputs):
        embedding_weights = self.embeds(inputs).view((-1,101)) #shape[1, 495]
        #self.attn_weights = torch.nn.Parameter(self.attn_weights)
        #attn_weights = autograd.Variable(torch.randn(self.embeds_size, self.embeds_size))
        #attn_weights = torch.nn.Parameter(attn_weights)       
        # transformation = self.tanh(torch.mm(embedding_weights, self.attn_weights)) #shape[1, 495]
        # transformation = nn.functional.softmax(transformation, dim=1) #
        # context = torch.nn.Parameter(transformation) # shape[1, 495]
        
#         print(embedding_weights.shape)
#         print(context.shape)
        
        #attended_inputs = embedding_weights * context # shape[1, 495]
        attended_inputs = embedding_weights# shape[1, 495]
        layer1 = self.fc1(attended_inputs) # shape[1,500]
        act1 = self.relu(layer1) # shape[1,500]
        #dout = self.dout(h1)
        layer2 = self.fc2(act1) # 
        act2 = self.relu(layer2)
        #layer3 = self.out(act2)
        #act3 = self.relu(layer3)
        #output = self.sigmoid(act3)
        output = self.sigmoid(act2)
        return output
    
    # def train(self, X, y):
    #     y_hat = self.forward(X)
    #     self.backward(X, y, y_hat)
    
    # def backward(self, X, y, y_hat):
