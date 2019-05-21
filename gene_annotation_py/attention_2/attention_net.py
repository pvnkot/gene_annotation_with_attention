import torch
from torch import nn
import torch
from torch import nn
import utils
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
import Config

class Attention_Net(nn.Module):
    def __init__(self, sent_size, device):
        super(Attention_Net, self).__init__()
        #embeds are of the dimension n * (1 x (embedding_size * no_of_kmers))
        self.embeds = nn.Embedding(len(utils.create_vocabulary(Config.window_size)), Config.embedding_size).to(device=device)
        #Embeds[125, 5]
        self.embeds_size = Config.embedding_size * sent_size
        self.sent_size = sent_size
        """
        Experimenting Start
        """
        #self.embeds_size = sent_size # set this if we want the embeddings to be fed as vectors
        self.attn_weights = nn.Parameter(autograd.Variable(torch.randn(self.embeds_size, self.embeds_size)))

        #self.attention = MultiHeadAttention(Config.n_head, Config.d_model, Config.d_k, Config.d_v, dropout=Config.attn_dropout);
        # self.attention = StructuredSelfAttn(self.sent_size, self.sent_size, Config.n_hops, Config.nlayers, Config.d_a, dropout=Config.attn_dropout);
        self.attention = StructuredSelfAttn(Config.embedding_size, Config.embedding_size, Config.n_hops, Config.nlayers, Config.d_a, dropout=Config.attn_dropout, device=device);
        """                                    
        Experimenting END
        """
        #self.attn_weights = nn.Parameter(autograd.Variable(torch.randn(self.embeds_size, self.embeds_size)))
        #self.attn_weights = autograd.Variable(torch.randn(self.embeds_size, self.embeds_size))
        #attn_weights = autograd.Variable(torch.randn(self.embeds_size, self.embeds_size))
        self.device = device
        self.tanh = torch.tanh
        self.fc1 = nn.Linear(self.embeds_size, Config.hidden_layer_size).to(device=device)
        """
        Experimenting Start
        """
        #self.fc1 = nn.Linear(Config.embedding_size, Config.hidden_layer_size)
        """
        Experimenting END
        """
        self.relu = F.relu
        #self.context = None
        self.sigmoid = nn.Sigmoid().to(device=device)
        self.fc2 = nn.Linear(Config.hidden_layer_size, 1).to(device=device)
        self.threshold = F.threshold
        self.dropout = nn.Dropout(Config.dropout).to(device=device)
        #self.out = nn.Linear(hidden_layer_size_2, 1)

    def forward(self, inputs):
        #embedding_weights = self.embeds(inputs).view((-1, self.embeds_size)) 
        inputs = inputs.to(device=self.device)
        """
        Experimenting Start
        """
        embedding_weights = self.embeds(inputs).view((-1, int(self.sent_size), Config.embedding_size))
        """
        Experimenting END
        """
        
        # attended_inputs = embedding_weights * context 
        attended_inputs = embedding_weights
        if Config.with_attention:
            attended_inputs = self.attention(attended_inputs);
        #attn_inputs = embedding_weights
        #print(attended_inputs)
        #attended_inputs = utils.apply_attention(self, self.attn_weights, embedding_weights)
            attended_inputs = attended_inputs.view(-1, self.embeds_size)
        layer1 = self.fc1(attended_inputs)
        layer1 = self.dropout(layer1)
        act1 = self.relu(layer1) 
        #act1 = self.threshold(layer1, 0.2, 0) 
        layer2 = self.fc2(act1) # 
        layer2 = self.dropout(layer2)
        act2 = self.relu(layer2)
        #act2 = self.threshold(layer2, 0.2, 0)
        """
        Experimenting Start
        """
        #output = self.sigmoid(torch.sum(act2, 1))
        """
        Experimenting END
        """
        output = self.sigmoid(act2)
        #output = self.softmax(act2)
        return output
    
class StructuredSelfAttn(nn.Module):
    ''' Structured Self-Attention module '''

    def __init__(self, ninp, nhid, n_hops, nlayers, d_a, dropout=0.1, device='cpu'):
        super().__init__()

        self.n_hops = n_hops
        self.d_a = d_a
        self.biLSTM = nn.LSTM(ninp, nhid, nlayers, dropout=Config.attn_dropout, bidirectional=True).to(device=device)
        self.w_s1 = nn.Linear(2*nhid, self.d_a, bias=False).to(device=device)# W_s1 of shape 2*u, d_a
        self.w_s2 = nn.Linear(self.d_a, self.n_hops, bias=False).to(device=device)# W_s2 of shape d_a, n_hops
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2).to(device=device) #Need to check the meaning of dim=2




        

    def forward(self, input_embeds, mask=None):

        d_a, n_hops = self.d_a, self.n_hops
        out, (h,c) = self.biLSTM(input_embeds)
        # Shape of out is: batch, num_directions(2 for biLSTM) * hidden_size
        size = out.size()
        compressed_embeddings = out.view(-1, size[2])  # [bsz*len, nhid*2]
        # transformed_inp = torch.transpose(input_embeds, 0, 1).contiguous()  # [bsz, len]
        # transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        # concatenated_inp = [transformed_inp for i in range(self.n_hops)]
        # concatenated_inp = torch.cat(concatenated_inp, 1) # [bsz, hop, len]
        
        first_term = self.w_s1(self.dropout(compressed_embeddings))
        hbar = self.tanh(first_term)
        alphas = self.w_s2(hbar).view(size[0], size[1], -1)
        
        
        #attended_inputs = self.softmax(attended_inputs)
        #print(next)
        alphas = self.softmax(alphas)
        
        attended_inputs = torch.mul(input_embeds, alphas)
        return attended_inputs
