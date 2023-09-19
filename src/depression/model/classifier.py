import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, num_phq_class=1, num_depression_class=2):
        super(Classifier, self).__init__()

        self.num_depression_class = 2

        self.depr = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim // 2),                    
            nn.BatchNorm1d(num_features=hidden_dim // 2),
            nn.ReLU(True), 
            nn.Dropout(dropout_rate), 
            nn.Linear(in_features=hidden_dim // 2, out_features=hidden_dim // 4), 
            nn.BatchNorm1d(num_features=hidden_dim // 4),
            nn.ReLU(True), 
            nn.Dropout(dropout_rate), 
            nn.Linear(in_features=hidden_dim // 4, out_features=num_depression_class)
        )

        self.score = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim // 4),                    
            nn.BatchNorm1d(num_features=hidden_dim // 4), 
            nn.ReLU(True), 
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=hidden_dim // 4, out_features=num_phq_class)
        )


    def forward(self, features):
        score_prob = self.score(features)
        depression_prob = F.one_hot((score_prob >= 10).int().view(-1).to(torch.long), num_classes=self.num_depression_class).to(torch.float)

        return depression_prob, score_prob