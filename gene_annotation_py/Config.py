import random

positive_sample_size = 100000
negative_sample_size = 100000
#positive_test_sample_size = 300
#negative_test_sample_size = 300
positive_train_sample_file = 'D1/positive_sample.txt'
negative_train_sample_file = 'D1/negative_sample.txt'
#positive_test_sample_file = 'D1/positive_sample_test.txt'
#negative_test_sample_file = 'D1/negative_sample_test.txt'
window_size = 3
embedding_size = 5
num_epochs = 100
batch_size = 100
hidden_layer_size = 99
hidden_layer_size_2 = 99
with_attention = True
learning_rate = 0.05
model_name = 'models/fc_with_attention.pt'
test_model_name = 'models/fc_with_attn_per_epoch.pt'
test_size = 0.25
seed = random.randint(1, 101)

#Attention config
mask = None
n_head = 1
d_model = 1
d_k = 16
d_v = 16
dropout = 0.1
