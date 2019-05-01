import random

positive_sample_size = 1000
negative_sample_size = 1000

positive_train_sample_file = 'D1/positive_sample.txt'
negative_train_sample_file = 'D1/negative_sample.txt'

window_size = 3
embedding_size = 5
num_epochs = 100
batch_size = 100
hidden_layer_size = 99
hidden_layer_size_2 = 99
with_attention = True
learning_rate = 0.1
model_name = 'models/fc_with_attention.pt'
test_model_name = 'models/fc_with_attn_per_epoch.pt'
test_size = 0.25
seed = random.randint(1, 101)
dropout = 0.02

#Attention config
mask = None
n_hops = embedding_size
d_a = 9
attn_dropout = 0.5
nlayers = 2
