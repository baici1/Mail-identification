import numpy as np
import os
from requests import get
import torch
from torch.utils.data import Dataset

from pre_processing import pre_processing


class GetData:
    def __init__(self, dir) -> None:
        with open(os.path.join(dir, "full/index"), "r") as f:
            index = np.array(f.readlines())
            data_path_list, label_list = zip(
                *[(x.split(" ")[1], x.split(" ")[0]) for x in index]
            )
            self.data_path_list = [
                x.replace("..", dir).replace("\n", "") for x in data_path_list
            ]
            self.label_list = [1 if x == "spam" else 0 for x in label_list]
            assert (
                type(self.data_path_list[0]) == str and type(self.label_list[0]) == int
            )

    def get_data_path_list(self) -> list:
        """return path of data

        Returns:
            list[str]: [N,]
        """
        return self.data_path_list

    def get_label_list(self) -> list:
        """return label

        Returns:
            list[int]: [N,]
        """
        return self.label_list


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

    def __getitem__(self, index) -> torch.tensor and torch.tensor:
        # train and vaild dataset pipeline
        if self.phase == "train":
            mail_token = pre_processing(
                self.data_path_list[index], self.pre_train_model, self.max_length
            )
            label = self.label_list[index]
        else:
            mail_token = pre_processing(
                self.data_path_list[index], self.pre_train_model, self.max_length
            )
            label = self.label_list[index]
        return mail_token, label

    def __len__(self) -> int:
        return len(self.label_list)


class TestDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()


get_data = GetData("trec06c")
train_data_set = TrainDataset(
    get_data.get_data_path_list, get_data.get_label_list, "train"
)
