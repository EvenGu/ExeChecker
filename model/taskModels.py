import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self, net, num_class=20):
        super(Classifier, self).__init__()
        self.net = net
        self.fcn = nn.Linear(net.out_channels, num_class)
        
    def forward(self, x):
        # N, C, T, V, M = x.shape
        x = self.net(x) # features
        x = self.fcn(x) 

        return x



class MultiTaskModel(nn.Module):
    def __init__(self, graph_net, embedding_dim=128):
        super(MultiTaskModel, self).__init__()
        self.graph_net = graph_net
        self.embedding = nn.Linear(graph_net.out_channels, embedding_dim)  
        
    def forward(self, x):
        features, _ = self.graph_net(x)
        embeddings = self.embedding(features)
        return features, embeddings

