import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable

from torch_geometric.nn import GCNConv, global_add_pool


class UtterNetwork(nn.Module):
    def __init__(
            self, 
            node_feature_dim, 
            hidden_channel_dim, 
            num_utter_window, 
            dropout_rate, 
            tokenizer, 
            device, 
            num_layers = 2, 
            bidirectional = True
        ) -> None:
        super(UtterNetwork, self).__init__()

        # Embedding
        vocab_size, embedding_dim = tokenizer.embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.init_pretrained_embeddings_from_numpy(tokenizer.embedding_matrix)

        # GRU
        if bidirectional: rnn_hidden_size = node_feature_dim // 2
        else: rnn_hidden_size = node_feature_dim

        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=rnn_hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

        # GNN
        self.node_feature_dim = node_feature_dim
        self.num_utter_window = num_utter_window

        self.gconv_1 = GCNConv(in_channels=node_feature_dim, out_channels=hidden_channel_dim, add_self_loops=False)
        self.gconv_2 = GCNConv(in_channels=hidden_channel_dim, out_channels=hidden_channel_dim, add_self_loops=False)
        self.gconv_3 = GCNConv(in_channels=hidden_channel_dim, out_channels=hidden_channel_dim, add_self_loops=False)

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.device = device

    def forward(self, input_ids, umasks, utter_lens):
        num_batch, num_utters, num_words = input_ids.size()

        embed_inputs = self.embedding(input_ids.to(torch.long))                                 # [batch_sie, num_utt, nun_words, embedding_dim]
        embed_inputs = embed_inputs.sum(dim=2)                                                  # [batch_size, num_utt, embedding_dim]

        mask = umasks.unsqueeze(-1).to(torch.float)                                             # [batch, num_utt, 1]
        mask = mask.repeat(1, 1, self.node_feature_dim)                                         # [batch, num_utt, feature_dim]

        utter_embeddings, _ = self.gru(embed_inputs)                                            # [batch, num_utt, feature_dim]
        utter_embeddings = (utter_embeddings * mask).reshape(-1, self.node_feature_dim)         # [batch * num_utt, feature_dim]

        gx, edge_index, batch = self.graphity(utter_embeddings, num_utters, utter_lens)
        gx = self.dropout_layer(torch.relu(self.gconv_1(gx, edge_index)))
        gx = self.dropout_layer(torch.relu(self.gconv_2(gx, edge_index)))
        gx = self.gconv_3(gx, edge_index)

        output = self.dropout_layer(global_add_pool(gx, batch))

        return gx, output

    def graphity(self, utter_emb, num_utters, utter_lens: Tensor):
        if isinstance(utter_lens, Tensor): utter_lens = utter_lens.detach().cpu().numpy().astype(int)

        gx = torch.zeros((np.sum(utter_lens), self.node_feature_dim))
        edge_index = list()
        batch = list()

        s_idx = 0
        for idx, u_len in enumerate(utter_lens):
            gx[s_idx: s_idx + u_len] = utter_emb[num_utters * idx: num_utters * idx + u_len]

            for src_index in range(u_len):
                for dst_index in range(max(0, src_index - self.num_utter_window), min(src_index + self.num_utter_window + 1, u_len)):
                    edge_index.append([src_index + s_idx, dst_index + s_idx])

            batch.extend([idx] * u_len)

            s_idx += u_len

        gx = Variable(gx, requires_grad=True).to(self.device)
        edge_index = Variable(torch.tensor(np.array(edge_index).T, dtype=torch.long), requires_grad=False).to(self.device)
        batch = Variable(torch.tensor(batch)).to(self.device)

        return gx, edge_index, batch
    
    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained_word_vectors))
        self.embedding.weight.requires_grad = False