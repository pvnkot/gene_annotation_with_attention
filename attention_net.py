
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
        self.embeds_size = Config.embedding_size*(kmer_size)
        #self.attn_weights = nn.Parameter(torch.randn(embeds_size, embeds_size))
        self.tanh = torch.tanh
        self.fc1 = nn.Linear(self.embeds_size, Config.hidden_layer_size)
        self.relu = F.relu
        self.sigmoid = nn.Sigmoid()
        #self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size_2)
        #self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size_2)
        self.fc2 = nn.Linear(Config.hidden_layer_size, 1)
        #self.out = nn.Linear(hidden_layer_size_2, 1)

    def forward(self, inputs):
        embedding_weights = self.embeds(inputs).view((1,-1))
        attn_weights = autograd.Variable(torch.randn(self.embeds_size, self.embeds_size))
        transformation = self.tanh(torch.mm(embedding_weights, attn_weights))
        context = self.sigmoid(transformation)
        
#         print(embedding_weights.shape)
#         print(context.shape)
        
        attended_inputs = embedding_weights * context
        layer1 = self.fc1(attended_inputs)
        act1 = self.relu(layer1)
        #dout = self.dout(h1)
        layer2 = self.fc2(act1)
        act2 = self.relu(layer2)
        #layer3 = self.out(act2)
        #act3 = self.relu(layer3)
        #output = self.sigmoid(act3)
        output = self.sigmoid(act2)
        return output
    