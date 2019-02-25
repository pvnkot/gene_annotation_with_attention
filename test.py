import torch
import attention_net as net
import Config
import utils

def test(inputs, labels, kmer_size, model_name):
    model = net.Attention_Net(kmer_size)
    model.load_state_dict(torch.load(Config.model_name))
    model.eval()
    labels_hat = []
    j=0
    vocabulary = utils.create_vocabulary(Config.window_size)
    for data in inputs.itertuples():
        gene = data.Gene
        input_ = torch.tensor([vocabulary[gene[i:i+Config.window_size]] for i in range(0, len(gene) - Config.window_size + 1)], dtype=torch.long)
        j += 1
        label_hat = model(input_)
        labels_hat.append(label_hat)
        
    accuracy = utils.get_test_accuracy(labels_hat)
    return accuracy