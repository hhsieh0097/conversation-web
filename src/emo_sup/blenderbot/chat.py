from deep_translator import GoogleTranslator

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from src.config import parse_args
from src.emo_sup.blenderbot.strat_blenderbot_small import Model


class EmoSupModel(object):
    def __init__(self) -> None:
        self.args = parse_args()

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.blenderbot_path)
        self.tokenizer.add_tokens(
            [
                "[Question]",
                "[Restatement or Paraphrasing]",
                "[Reflection of feelings]",
                "[Self-disclosure]",
                "[Affirmation and Reassurance]",
                "[Providing Suggestions]",
                "[Information]",
                "[Others]"
            ]
        )

        self.model = Model.from_pretrained(self.args.blenderbot_path)
        self.model.tie_tokenizer(self.tokenizer)

        self.model.load_state_dict(torch.load(self.args.blenderbot_ckpt_path))
        self.model.eval()

        self.generation_kwargs = {
            'max_length': 40,
            'min_length': 10,
            'do_sample': False,
            'temperature': 0.7,
            'top_k': 0,
            'top_p': 0.9,
            'num_beams': 1,
            'num_return_sequences': 1,
            'length_penalty': 1.0,
            'repetition_penalty': 1.0,
            'no_repeat_ngram_size': 0,
            'encoder_no_repeat_ngram_size': 0, 
            'pad_token_id': 0,
            'bos_token_id': 1,
            'eos_token_id': 2,
        }
        
        self.trans_zh2en = GoogleTranslator(source='zh-TW', target='en')
        self.trans_en2zh = GoogleTranslator(source='en', target='zh-TW')

    
    def preprocess_conversation(self, context, max_turn):
        _context = context.copy()
        if max_turn == -1: pass
        else:
            _context = _context[-(max_turn * 2) + 1: ]

        _context_en = self.trans_zh2en.translate_batch([v for sent in _context for k, v in sent.items()])
        _context_en = [{k: _context_en[idx]} for idx, sent in enumerate(_context) for k, v in sent.items()]

        return _context_en


    def postprocess_response(self, response):
        response_zh = self.trans_en2zh.translate(response)

        return response_zh

    
    def chatting(self, context):
        processed_context = self.preprocess_conversation(context, self.args.max_turn)

        infer_dataset = BlenderDataset([processed_context], self.tokenizer)
        infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, collate_fn=infer_dataset.collect_fn)

        for batch in infer_dataloader:
            batch.update(self.generation_kwargs)

            with torch.no_grad():
                encoded_info, generations = self.model.generate(**batch)

        assistant_response = self.tokenizer.decode(generations[0], skip_special_tokens=True)
        assistant_response = self.postprocess_response(assistant_response)

        return assistant_response


class BlenderDataset(Dataset):
    def __init__(self, dialog, tokenizer) -> None:
        super().__init__()

        self.dialog = dialog
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dialog)
    
    def __getitem__(self, index):
        return self.dialog[index]
    
    def collect_fn(self, data):
        for inst in data:
            bot_input = [v + ' __end__ ' for sent in inst for k, v in sent.items()]
            bot_input = ''.join(bot_input)

        encoded = self.tokenizer(
            bot_input, 
            padding='max_length', 
            max_length=160, 
            truncation=True, 
            return_tensors='pt'
        )
        input_ids, attention_mask = encoded['input_ids'], encoded['attention_mask']
        
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'decoder_input_ids': torch.tensor([[1]])
        }