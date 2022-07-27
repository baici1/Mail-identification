from typing import OrderedDict
import numpy as np
import os
from requests import get
import torch
from torch.utils.data import Dataset

from pre_processing import pre_processing


def get_data(data_dir) -> list(tuple()):
    """return the paths and labels of data

    Args:
        data_dir (str): the dir of data

    Returns:
        list[tuple]: the paths and labels of data, [N, 2],
        the first dim is path of data, the second dim is label
    """
    with open(os.path.join(data_dir, "full/index"), "r") as f:
        index = np.array(f.readlines())
        data_path_list, label_list = zip(
            *[(x.split(" ")[1], x.split(" ")[0]) for x in index]
        )
        data_path_list = [
            x.replace("..", data_dir).replace("\n", "") for x in data_path_list
        ]
        label_list = [1 if x == "spam" else 0 for x in label_list]
        assert type(data_path_list[0]) == str and type(label_list[0]) == int
        return list(zip(data_path_list, label_list))


class TrainDataset(Dataset):
    def __init__(
        self,
        data_path_list,
        label_list,
        pre_train_model="chinese-bert-wwm",
        max_length=512,
        phase="train",
    ) -> None:
        super().__init__()
        assert phase in ["train", "valid"]
        self.data_path_list = data_path_list
        self.label_list = np.array(label_list)
        self.pre_train_model = pre_train_model
        self.max_length = max_length
        self.phase = phase

    def __getitem__(self, index) -> tuple:
        # train and vaild dataset pipeline
        with open(
            self.data_path_list[index], "r", encoding="gb18030", errors="ignore"
        ) as f:
            mail = "".join(f.readlines())
        if self.phase == "train":
            mail_token = pre_processing(mail, self.pre_train_model, self.max_length)
            label = self.label_list[index]
        else:
            mail_token = pre_processing(mail, self.pre_train_model, self.max_length)
            label = self.label_list[index]
        input_ids = mail_token["input_ids"].squeeze(0)
        attention_mask = mail_token["attention_mask"].squeeze(0)
        token_type_ids = mail_token["token_type_ids"].squeeze(0)
        return input_ids, attention_mask, token_type_ids, label

    def __len__(self) -> int:
        return len(self.label_list)


class TestDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
