import os
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

from src.config import parse_args
from src.emotion.emo_detec import VAD
from src.depression.tokenizer.tokenizer import MyTokenizer
from src.depression.model.hierarchical import HierDeprDetec


class PersonalDetection(object):
    def __init__(self, vad_trained_path, detec_trained_path) -> None:
        self.args = parse_args()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._VAD = VAD(pretrained_model=vad_trained_path)

        self.tokenizer = MyTokenizer(trained_cache=detec_trained_path)
        self.model = HierDeprDetec(
            token_node_feature_dim=self.tokenizer.embedding_matrix.shape[1], 
            token_hidden_channel_dim=self.args.token_hidden_channel_dim, 
            utter_node_feature_dim=self.args.utter_node_feature_dim, 
            utter_hidden_channel_dim=self.args.utter_hidden_channel_dim, 
            num_utter_window=self.args.num_utter_window, 
            emotion_class=self.args.emotion_class, 
            ea_head=self.args.ea_head, 
            hidden_dim=self.args.hidden_dim, 
            dropout_rate=self.args.dropout_rate, 
            tokenizer=self.tokenizer, 
            device=self.device
        )
        
        self.model.load_state_dict(torch.load(os.path.join(detec_trained_path, 'best.pt')))
        self.model = self.model.to(self.device)

    def get_depression(self, context, gender):
        personal_data = [self._VAD.get_vad(context)]
        personal_data[0]['gender'] = 0 if gender == 'Male' else 1

        _graph = DeprGraph(self.tokenizer, n_gram=self.args.n_gram)
        personal_data = _graph.graph_builder(personal_data)

        _dataset = DeprDataset(data=personal_data, tokenizer=self.tokenizer, max_token_len=self.args.max_token_len)
        _dataloader = DataLoader(_dataset, batch_size=1, collate_fn=_dataset.collate_fn, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            for _, batch in enumerate(_dataloader):
                batch = {level_k: {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in v_k.items()} for level_k, v_k in batch.items()}
                
                logits = self.model(**batch)
                depression = torch.argmax(logits['depressions']).item()

        return round(logits['scores'].item(), 4), depression


class DeprGraph():
    def __init__(self, tokenizer, n_gram: int):
        super().__init__()
        
        self.n_gram = n_gram
        self.tokenizer = tokenizer

    def build_token_nodes(self, transcript):
        node_vocab = list()

        for utterance in transcript:
            for vocab in utterance.split(' '):
                node_vocab.append(vocab)

        return list(set(node_vocab))

    def build_token_edges(self, transcript, nodes):
        edges = list()
        word_count = np.zeros(len(nodes))
        pair_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

        for utterance in transcript:
            tokens = utterance.split()

            for index, src_word in enumerate(tokens):
                src_index = nodes.index(src_word)
                word_count[src_index] += 1

                for idx in range(max(0, index - self.n_gram), min(index + self.n_gram + 1, len(tokens))):
                    dst_word = tokens[idx]
                    dst_index = nodes.index(dst_word)
                    if [src_index, dst_index] not in edges:
                        edges.append([src_index, dst_index])

                    pair_matrix[src_index, dst_index] += 1

        total = np.sum(word_count)
        word_count = word_count / total
        pair_matrix = pair_matrix / total

        pmi_matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
        for row in range(len(nodes)):
            for col in range(len(nodes)):
                pmi_matrix[row, col] = np.log(
                    np.divide(pair_matrix[row, col], 
                              (word_count[row] * word_count[col]), 
                              out=np.zeros_like(pair_matrix[row, col]), 
                              where=(word_count[row] * word_count[col]) != 0) 
                    + 1e-10
                )

                if pmi_matrix[row, col] <= 0:
                    pmi_matrix[row, col] = 0.0

        edge_weight = list()
        for e in edges:
            src, dst = e[0], e[1]
            edge_weight.append(pmi_matrix[src, dst])
        edge_weight = np.array(edge_weight).reshape(-1, 1)

        return edges, edge_weight

    def extract_token_embed(self, nodes):
        feature_matrix = list()

        for token in nodes:
            feature_matrix.append(self.tokenizer.embedding_matrix[self.tokenizer.token2idx.get(token, 0)])

        return np.array(feature_matrix)

    def build_token_graph(self, text_data):
        nodes = self.build_token_nodes(text_data)
        edges, edge_weight = self.build_token_edges(text_data, nodes)
        
        node_features = self.extract_token_embed(nodes)
        edges = [np.array([edge[0], edge[1]]) for edge in edges]
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        edge_weight = torch.as_tensor(edge_weight, dtype=torch.float)

        f = torch.tensor(node_features, dtype=torch.float32)
        graph_data = Data(x=f, edge_index=edge_index, edge_attr=edge_weight)

        return graph_data
    
    def graph_builder(self, data):
        for idx in range(len(data)):
            instance = data[idx]
            
            transcripts = instance['transcripts']
            token_graph_data = self.build_token_graph(transcripts)
            instance['token_graph'] = token_graph_data

        graph_list = [data[idx]['token_graph'] for idx in range(len(data))]

        return data
    

class DeprDataset(Dataset):
    def __init__(
            self, 
            data,
            tokenizer, 
            max_token_len: int, 
        ):

        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        instance = self.data[index]

        transcripts, valence, arousal, dominance, gender, portion_neg, token_graph = instance['transcripts'], \
                                                                                     instance['valence'], \
                                                                                     instance['arousal'], \
                                                                                     instance['dominance'], \
                                                                                     instance['portion_neg'], \
                                                                                     instance['gender'], \
                                                                                     instance['token_graph']

        encoded = list()
        for t_idx in range(len(transcripts)):
            utterance = transcripts[t_idx]
            encoded.append(self.tokenizer.encode(utterance, padding='max_length', truncation=True, max_length=self.max_token_len))

        num_utters = len(encoded)

        instance = {
            'token_level': {
                'graph': token_graph
            }, 
            'utter_level':  {
                'input_ids': encoded, 
                'num_utters': num_utters
            }, 
            'labels': {
                'gender': gender, 
                'valence': valence, 
                'arousal': arousal, 
                'dominance': dominance, 
                'portion_neg': portion_neg
            }
        }

        return instance
    
    def collate_fn(self, data):
        batch_info = {
            'token_level': {'graphs': list()}, 
            'utter_level': {'input_ids': list(), 'umasks': list(), 'utter_lens': list()}, 
            'labels': {'genders': list(), 'valences': list(), 'arousals': list(), 'dominances': list(), 'portion_negs': list()}
        }

        max_num_utters = max([data[idx]['utter_level']['num_utters'] for idx in range(len(data))])

        for idx in range(len(data)):
            token_level, utter_level, labels = data[idx].values()

            batch_info['token_level']['graphs'].append(token_level['graph'])

            pad_utters = [([self.tokenizer.pad_token_id] * self.max_token_len)] * (max_num_utters - utter_level['num_utters'])
            batch_info['utter_level']['input_ids'].append(utter_level['input_ids'] + pad_utters)
            batch_info['utter_level']['umasks'].append([1] * utter_level['num_utters'] + [0] * (max_num_utters - utter_level['num_utters']))
            batch_info['utter_level']['utter_lens'].append(utter_level['num_utters'])

            batch_info['labels']['genders'].append([labels['gender']])

            batch_info['labels']['valences'].append([labels['valence']])
            batch_info['labels']['arousals'].append([labels['arousal']])
            batch_info['labels']['dominances'].append([labels['dominance']])
            batch_info['labels']['portion_negs'].append([labels['portion_neg']])

        batch_info['utter_level'] = {k: torch.FloatTensor(v) for k, v in batch_info['utter_level'].items()}
        batch_info['labels'] = {k: torch.LongTensor(v) if k not in ['genders', 'valences', 'arousals', 'dominances', 'portion_negs'] else torch.FloatTensor(v) for k, v in batch_info['labels'].items()}

        return batch_info