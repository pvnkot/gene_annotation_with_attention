import torch
from torch import nn
import torch
from torch import nn
import utils
import torch.nn.functional as F
import torch.autograd as autograd
import Config

class Attention_Net(nn.Module):
    def __init__(self, sent_size):
        super(Attention_Net, self).__init__()
        #embeds are of the dimension n * (1 x (embedding_size * no_of_kmers))
        self.embeds = nn.Embedding(len(utils.create_vocabulary(Config.window_size)), Config.embedding_size)
        #Embeds[125, 5]
        self.embeds_size = Config.embedding_size * sent_size
        """
        Experimenting Start
        """
        #self.embeds_size = sent_size # set this if we want the embeddings to be fed as vectors
        """
        Experimenting END
        """
        #self.attn_weights = nn.Parameter(autograd.Variable(torch.randn(self.embeds_size, self.embeds_size)))
        #self.attn_weights = autograd.Variable(torch.randn(self.embeds_size, self.embeds_size))
        #attn_weights = autograd.Variable(torch.randn(self.embeds_size, self.embeds_size))
        self.tanh = torch.tanh
        self.fc1 = nn.Linear(self.embeds_size, Config.hidden_layer_size)
        """
        Experimenting Start
        """
        #self.fc1 = nn.Linear(Config.embedding_size, Config.hidden_layer_size)
        """
        Experimenting END
        """
        self.relu = F.relu
        #self.context = None
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(Config.hidden_layer_size, 1)
        #self.out = nn.Linear(hidden_layer_size_2, 1)

    def forward(self, inputs):
        embedding_weights = self.embeds(inputs).view((-1, self.embeds_size)) 
        """
        Experimenting Start
        """
        #embedding_weights = self.embeds(inputs).view((-1, self.embeds_size, Config.embedding_size))
        """
        Experimenting END
        """
        #self.attn_weights = torch.nn.Parameter(self.attn_weights)
        #attn_weights = autograd.Variable(torch.randn(self.embeds_size, self.embeds_size))
        #attn_weights = torch.nn.Parameter(attn_weights)       
        # transformation = self.tanh(torch.mm(embedding_weights, self.attn_weights)) 
        # transformation = nn.functional.softmax(transformation, dim=1) 
        # context = torch.nn.Parameter(transformation) 
        
        # print(embedding_weights.shape)
        # print(context.shape)
        
        # attended_inputs = embedding_weights * context 
        attended_inputs = embedding_weights
        layer1 = self.fc1(attended_inputs) 
        act1 = self.relu(layer1) 
        #act1 = self.sigmoid(layer1) 
        layer2 = self.fc2(act1) # 
        act2 = self.relu(layer2)
        #act2 = self.sigmoid(layer2)
        """
        Experimenting Start
        """
        #output = self.sigmoid(torch.sum(act2, 1))
        """
        Experimenting END
        """
        output = self.sigmoid(act2)
        return output
    
    # def train(self, X, y):
    #     y_hat = self.forward(X)
    #     self.backward(X, y, y_hat)
    
    # def backward(self, X, y, y_hat):
