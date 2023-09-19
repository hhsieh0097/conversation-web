import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaConfig

from src.emotion.vad_extraction import PretrainedLMModel


class VAD(object):
    def __init__(self, pretrained_model) -> None:

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        config = RobertaConfig.from_pretrained(pretrained_model_name_or_path='roberta-base')
        state = torch.load(pretrained_model)
        
        self.model = PretrainedLMModel(config, model_name='roberta-base')
        self.model.load_state_dict(state['state_dict'], strict=False)

        self.model.to(self.device)
    
    def get_vad(self, context):
        vad_list = list()
        valence, arousal, dominance, portion_neg = -1, -1, -1, -1

        _dataset = VADDataset(context, self.tokenizer)
        _data_loader = DataLoader(_dataset, batch_size=32, shuffle=False)

        for idx, batch in enumerate(_data_loader):
            batch = {k: v.squeeze(1).to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                _, cls_logits = self.model(**batch)

            vad_list.extend(F.relu(cls_logits).cpu().detach().numpy().tolist())

        mean_vad = np.array(vad_list).mean(axis=0).round(4).tolist()
        valence, arousal, dominance = [value for value in mean_vad]

        each_v = [vad[0] for vad in vad_list]
        each_v_neg = [1 if v <= 2.5 else 0 for v in each_v]
        portion_neg = np.array(each_v_neg).mean().round(4)

        print(
            {
                'transcripts': context, 
                'valence': valence, 
                'arousal': arousal, 
                'dominance': dominance, 
                'portion_neg': portion_neg
            }
        )

        return {
            'transcripts': context, 
            'valence': valence, 
            'arousal': arousal, 
            'dominance': dominance, 
            'portion_neg': portion_neg
        }


class VADDataset(Dataset):
    def __init__(self, transcripts, tokenizer) -> None:
        self.transcripts = transcripts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.transcripts)
    
    def __getitem__(self, index):
        encoded_data = self.tokenizer(
            self.transcripts[index], 
            max_length=256, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )

        return encoded_data