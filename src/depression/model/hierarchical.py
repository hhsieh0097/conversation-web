import torch
import torch.nn as nn

from src.depression.model.token_extraction import TokenNetwork
from src.depression.model.utter_extraction import UtterNetwork
from src.depression.model.attention import ExternalAttention
from src.depression.model.classifier import Classifier


class HierDeprDetec(nn.Module):
    def __init__(
            self, 
            token_node_feature_dim, 
            token_hidden_channel_dim, 
            utter_node_feature_dim, 
            utter_hidden_channel_dim, 
            num_utter_window, 
            emotion_class, 
            ea_head, 
            hidden_dim, 
            dropout_rate, 
            tokenizer, 
            device, 
        ) -> None:
        super(HierDeprDetec, self).__init__()

        self.token_network = TokenNetwork(
            node_feature_dim=token_node_feature_dim, 
            hidden_channel_dim=token_hidden_channel_dim, 
            dropout_rate=dropout_rate, 
            device=device
        )

        self.utter_network = UtterNetwork(
            node_feature_dim=utter_node_feature_dim, 
            hidden_channel_dim=utter_hidden_channel_dim, 
            num_utter_window=num_utter_window, 
            dropout_rate=dropout_rate, 
            tokenizer=tokenizer, 
            device=device
        )

        input_dim = token_hidden_channel_dim + utter_hidden_channel_dim + 1 + len(emotion_class) + 1  # Gender, Valence, Arousal and Dominance, Portion_neg
        
        self.emotion_class = emotion_class
        self.attention = ExternalAttention(d_model=input_dim, S=ea_head)

        self.classifier = Classifier(input_dim, hidden_dim, dropout_rate)

        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)

        self.apply(self.init_weight)

    def forward(
            self, 
            token_level = None, 
            utter_level = None, 
            labels = None
        ):

        genders = labels.get('genders')

        for idx in range(len((self.emotion_class))):
            if idx == 0: emotion = labels.get(self.emotion_class[idx])
            else: emotion = torch.cat((emotion, labels.get(self.emotion_class[idx])), dim=1)

        portion_negs = labels.get('portion_negs')

        token_x, token_features = self.token_network(**token_level)
        utter_x, utter_features = self.utter_network(**utter_level)

        personal_features = torch.cat((genders, emotion, portion_negs), dim=1)

        features = torch.cat((token_features, utter_features), dim=1)
        features = torch.cat((features, personal_features), dim=1)

        features = features.unsqueeze(1)
        features_attn = self.attention(features)
        features = (features + features_attn).squeeze(1)
        features = self.batch_norm(features)
        features = torch.relu(features)

        logits_depr, logits_score = self.classifier(features)

        return {
            'depressions': logits_depr, 
            'scores': logits_score
        }

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.01)