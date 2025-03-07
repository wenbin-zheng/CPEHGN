import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from transformers import BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class GraphConvolutionalStack(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1):
        super(GraphConvolutionalStack, self).__init__()
        self.layer_stack = nn.ModuleList([GraphConvolutionalLayer(input_dim if i == 0 else output_dim, output_dim)
                                          for i in range(num_layers)])

    def forward(self, node_features, adjacency_matrix):
        for layer in self.layer_stack:
            node_features = layer(node_features, adjacency_matrix)
        return node_features

class GraphConvolutionalLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionalLayer, self).__init__()
        self.linear_transform = nn.Linear(input_dim, output_dim)

    def forward(self, node_features, adjacency_matrix):
        graph_output = torch.spmm(adjacency_matrix, node_features)  # Matrix multiplication with adjacency
        graph_output = self.linear_transform(graph_output)
        return F.relu(graph_output)

class SelfAttention(nn.Module):
    def __init__(self, dimension):
        super(SelfAttention, self).__init__()
        self.scale_factor = dimension ** -0.5
        self.softmax_function = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale_factor
        attention_weights = self.softmax_function(attention_scores)
        return torch.matmul(attention_weights, value), attention_weights

class MultimodalFusion(nn.Module):
    def __init__(self, config, shared_dim=128, similarity_dim=64, use_text_graph=True, use_image_graph=True, use_cross_graph=True):
        super(MultimodalFusion, self).__init__()
        self.config = config
        self.num_events = config.event_num
        self.use_text_graph = use_text_graph
        self.use_image_graph = use_image_graph
        self.use_cross_graph = use_cross_graph
        num_classes = config.class_num
        self.hidden_dim = config.hidden_dim
        self.lstm_dim = config.embed_dim
        self.social_features = 19
        self.text_gcn = GraphConvolutionalStack(similarity_dim, similarity_dim, num_layers=2)
        self.image_gcn = GraphConvolutionalStack(similarity_dim, similarity_dim, num_layers=2)

        # BERT Model
        if self.config.dataset == 'weibo':
            bert_model = BertModel.from_pretrained('../chinese')
        else:
            bert_model = BertModel.from_pretrained('../uncased')

        self.bert_hidden_dim = config.bert_hidden_dim
        self.text_shared_linear = nn.Sequential(
            nn.Linear(self.bert_hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(self.bert_hidden_dim, self.hidden_dim)
        self.bertModel = bert_model
        self.dropout_layer = nn.Dropout(config.dropout)

        # IMAGE
        resnet = torchvision.models.resnet34(pretrained=True)
        num_ftrs = resnet.fc.out_features
        self.visual_model = resnet
        self.image_shared_layer = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )

        # Attention mechanism
        self.text_attention_layer = SelfAttention(shared_dim)

        # Fusion Layer
        self.text_fusion_layer = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, similarity_dim),
            nn.BatchNorm1d(similarity_dim),
            nn.ReLU()
        )

        self.image_fusion_layer = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, similarity_dim),
            nn.BatchNorm1d(similarity_dim),
            nn.ReLU()
        )

        self.image_fc = nn.Linear(num_ftrs, self.hidden_dim)

        # Classifier
        self.classification_layer = nn.Sequential()
        self.classification_layer.add_module('fc1', nn.Linear(2 * self.hidden_dim, 2))

        self.unimodal_classifier = nn.Sequential(
            nn.Linear(similarity_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax()
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(similarity_dim * 2, similarity_dim * 2),
            nn.BatchNorm1d(similarity_dim * 2),
            nn.ReLU(),
            nn.Linear(similarity_dim * 2, similarity_dim),
            nn.ReLU()
        )

        self.similarity_classifier = nn.Sequential(
            nn.Linear(similarity_dim * 3, similarity_dim),
            nn.BatchNorm1d(similarity_dim),
            nn.ReLU(),
            nn.Linear(similarity_dim, 2)
        )

    def initialize_hidden(self, batch_size):
        return (to_var(torch.zeros(1, batch_size, self.lstm_dim)),
                to_var(torch.zeros(1, batch_size, self.lstm_dim)))

    def convolution_and_pool(self, features, conv_layer):
        features = F.relu(conv_layer(features)).squeeze(3)
        features = F.max_pool1d(features, features.size(2)).squeeze(2)
        return features

    def construct_graph(self, text_features, image_features, intra_threshold=0.7, cross_threshold=0.7):
        # Calculate cosine similarity between text features
        text_similarity = cosine_similarity(text_features.detach().cpu().numpy())
        text_adj_matrix = (text_similarity >= intra_threshold).astype(float)

        # Calculate cosine similarity between image features
        image_similarity = cosine_similarity(image_features.detach().cpu().numpy())
        image_adj_matrix = (image_similarity >= intra_threshold).astype(float)

        # Calculate cosine similarity between text and image features
        cross_similarity = cosine_similarity(text_features.detach().cpu().numpy(), image_features.detach().cpu().numpy())
        cross_adj_matrix = (cross_similarity >= cross_threshold).astype(float)

        # Combine adjacency matrices
        combined_adj_matrix = np.maximum(text_adj_matrix, np.maximum(image_adj_matrix, cross_adj_matrix))
        return torch.FloatTensor(combined_adj_matrix).to(text_features.device)

    def forward(self, text_input, image_input, attention_mask):
        # IMAGE
        image_features = self.visual_model(image_input)
        image_z = self.image_shared_layer(image_features)
        image_z = self.image_fusion_layer(image_z)
        image_prediction = self.unimodal_classifier(image_z)

        last_hidden_state = torch.mean(self.bertModel(text_input)[0], dim=1, keepdim=False)
        text_z, _ = self.text_attention_layer(last_hidden_state, last_hidden_state, last_hidden_state)

        text_z = self.text_shared_linear(text_z)
        text_z = self.text_fusion_layer(text_z)

        text_prediction = self.unimodal_classifier(text_z)

        # Construct adjacency matrix
        text_adj_matrix = self.construct_graph(text_z, image_z)
        image_adj_matrix = self.construct_graph(image_z, text_z)

        #Modalities-Specific GCNs to enhance features
        enhanced_text_z = self.text_gcn(text_z, text_adj_matrix)
        enhanced_image_z = self.image_gcn(image_z, image_adj_matrix)

        image_alpha = image_prediction[:, -1] / (image_prediction[:, -1] + text_prediction[:, -1])
        text_alpha = text_prediction[:, -1] / (image_prediction[:, -1] + text_prediction[:, -1])

        image_alpha = image_alpha.unsqueeze(1)
        text_alpha = text_alpha.unsqueeze(1)

        text_image_features = torch.cat(((text_alpha * text_z), (image_alpha * image_z)), 1)
        text_image_features = self.fusion_layer(text_image_features)

        text_image_fusion = torch.cat((enhanced_text_z, text_image_features, enhanced_image_z), 1)

        # Final classification: Fake or Real
        final_output = self.similarity_classifier(text_image_fusion)

        final_output = self.dropout_layer(final_output)
        return final_output, image_prediction, text_prediction, text_image_features, image_z, text_z

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)