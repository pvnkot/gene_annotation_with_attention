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
        self.attn_weights = nn.Parameter(autograd.Variable(torch.randn(self.embeds_size, self.embeds_size)))

        self.attention = MultiHeadAttention(Config.n_head, Config.d_model, Config.d_k, Config.d_v, dropout=Config.dropout);
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
        self.threshold = F.threshold
        self.dropout = nn.Dropout(0.1)
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
        # attended_inputs = embedding_weights
        
        attended_inputs = self.attention(embedding_weights, embedding_weights, embedding_weights, mask=Config.mask);
        #attn_inputs = embedding_weights
        #print(attended_inputs)
        #attended_inputs = utils.apply_attention(self, self.attn_weights, embedding_weights)
        attn_inputs = attended_inputs[0].view(-1, 495)
        layer1 = self.fc1(attn_inputs) 
        act1 = self.relu(layer1) 
        #act1 = self.threshold(layer1, 0.2, 0) 
        layer2 = self.fc2(act1) # 
        #layer2 = self.dropout(layer2)
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
    
    # def train(self, X, y):
    #     y_hat = self.forward(X)
    #     self.backward(X, y, y_hat)
    
    # def backward(self, X, y, y_hat):

class DotProductAttention(nn.Module):
    def __init__(self, scaling_factor, attn_dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.scaling_factor = scaling_factor #supposed to be sqrt(d_k)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2) #Need to check the meaning of dim=2
    
    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.scaling_factor

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = DotProductAttention(scaling_factor=np.power(d_k, 0.5))
        #self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q = q.size()
        sz_b, len_k = k.size()
        sz_b, len_v = v.size()

        q = q.view(-1, 495, 1);
        k = k.view(-1, 495, 1);
        v = v.view(-1, 495, 1);

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        #mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        #mask = None
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        #output = self.layer_norm(output + residual)
        #
        # output = output + residual

        return output, attn
