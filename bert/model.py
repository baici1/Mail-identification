import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from pre_processing import pre_processing


class Bert(nn.Module):
    def __init__(self, classific_num=2):
        super().__init__()
        self.classific_num = classific_num
        config = BertConfig.from_pretrained("chinese-bert-wwm")
        self.backbone = BertModel.from_pretrained("chinese-bert-wwm", config=config)
        # fin-tune
        for param in self.backbone.parameters():
            param.requires_grad = True
        pooler_fc_size = config.to_dict()["pooler_fc_size"]
        self.fc = nn.Linear(pooler_fc_size, classific_num)

    def forward(self, x):
        """return the logits after the sigmod layer

        Args:
            x (dir {str:tensor}): the input of BertModel

        Returns:
            tensor: [B, classific_num]
        """
        cls = self.backbone(**x).pooler_output
        out = self.fc(cls.squeeze(0))
        # Binaryclassification(classific_num=2)
        if self.classific_num == 2:
            return torch.sigmoid(out)
        # Multiclassification(classific_num>2)
        else:
            return out


# model = Bert(classific_num=2)
# # print(model.parameters)
# model.eval()
# output = model(pre_processing("trec06c/data/000/011", 512))
# print(output)
