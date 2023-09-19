import torch.nn as nn

from transformers import BertPreTrainedModel, RobertaModel


class PretrainedLMModel(BertPreTrainedModel):
    def __init__(self, config, model_name):
        super(PretrainedLMModel, self).__init__(config)
        self.config = config

        self.pre_trained_lm = RobertaModel.from_pretrained(
            model_name,
            config=self.config
        )

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.projection_lm = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.activation = nn.Sigmoid()

        self.label_num = 1
        
        self.head = nn.Linear(self.config.hidden_size, self.label_num * 3)

    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            token_type_ids = None,
            position_ids = None,
            head_mask = None,
            inputs_embeds = None,
        ):

        lm_outputs = self.pre_trained_lm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=False,
        )

        hidden_states, pooled_output = lm_outputs

        lm_logits = self.projection_lm(hidden_states)

        pooled_output = self.dropout(pooled_output)
        logits = self.head(pooled_output)

        return lm_logits, logits