import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class Bert(nn.Module):
    def __init__(
        self, pre_train_model="chinese-bert-wwm", classific_num=2, resume=False
    ):
        super().__init__()
        self.classific_num = classific_num
        config = BertConfig.from_pretrained(pre_train_model)
        # Avoid renewal training to override model weights
        if resume:
            self.backbone = BertModel(config=config)
        else:
            self.backbone = BertModel.from_pretrained(pre_train_model, config=config)
        # Fine-tune
        for param in self.backbone.parameters():
            param.requires_grad = True
        pooler_fc_size = config.num_labels
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.fc = nn.Linear(pooler_fc_size, classific_num)

    def forward(self, x):
        """return the logits after the sigmod layer

        Args:
            x (dir {str:tensor}): the input of BertModel

        Returns:
            tensor: [B, classific_num]
        """
        cls = self.backbone(**x).pooler_output
        out = self.fc(self.dropout(cls.squeeze(0)))
        return out
