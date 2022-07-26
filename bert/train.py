from collections import OrderedDict
from glob import glob
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch import dtype, optim
from torch.utils.data import DataLoader
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import TrainDataset, get_data
from model import Bert

# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int)
# args = parser.parse_args()
# torch.cuda.set_device(args.local_rank)


class Train:
    def __init__(self, config) -> None:
        self.pre_train_model = config["pre_train_model"]
        self.data_dir = config["data_dir"]
        self.result_path = config["result_path"]
        self.batch_size = config["batch_size"]
        self.max_length = config["max_length"]
        self.num_epochs = config["num_epochs"]
        self.classific_num = config["classific_num"]

        self.num_workers = config["num_workers"]
        self.pin_memory = config["pin_memory"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.tensorboard_step = config["tensorboard_step"]
        self.device = config["device"]
        self.resume = config["resume"]

        self.writer = SummaryWriter(os.path.join(self.path + "log"))
        self.train_indices = OrderedDict()
        self.valid_indices = OrderedDict()
        self.start_epoch = 0
        self.forward_step = 0

    def setup_seed(self) -> None:
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def dataload(self) -> None:
        data_path_and_label_list = get_data(self.data_dir)
        train_data_path_and_label_list, vaild_data_path_and_label_list = self.split(
            data_path_and_label_list, 0.1
        )
        self.train_iter = DataLoader(
            dataset=TrainDataset(
                *zip(train_data_path_and_label_list),
                self.pre_train_model,
                self.max_length,
                "train"
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.valid_iter = DataLoader(
            dataset=TrainDataset(
                *zip(vaild_data_path_and_label_list),
                self.pre_train_model,
                self.max_length,
                "valid"
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def build_model(self) -> None:
        self.model = Bert(self.pre_train_model, self.classific_num, self.resume)

    def define_loss(self) -> None:
        if self.classific_num == 2:
            self.cls_loss = nn.BCELoss()
        else:
            self.cls_loss = nn.CrossEntropyLoss()

    def define_optim(self) -> None:
        self.optim = optim.AdamW(
            [
                {"params": self.model.backbone.parameters()},
                {"params": self.model.dropout.parameters()},
                {"params": self.model.fc.parameters(), "lr": self.lr["fc"]},
            ],
            lr=self.lr["backbone"],
            weight_decay=self.weight_decay,
        )

    def save_model(self, path, step) -> None:
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        params = {}
        params["model"] = self.model.state_dict()
        params["optim"] = self.optim.state_dict()
        params["train_indices"] = self.train_indices
        params["valid_indices"] = self.valid_indices
        params["start_epoch"] = self.start_epoch
        params["forward_step"] = self.forward_step
        torch.save(
            params,
            os.path.join(path, "model_params_%07d.pt" % step),
        )

    def load_model(self, path, step) -> None:
        params = torch.load(path, "model_params_%07d.pt" % step)
        self.model.load_state_dict(params["model"])
        self.optim.load_state_dict(params["optim"])
        self.train_indices = params["train_indices"]
        self.valid_indices = params["valid_indices"]
        self.start_epoch = params["start_epoch"]
        self.forward_step = params["forward_step"]
        # Restore tensorboard
        for key, value in self.train_indices.items():
            self.writer.add_scalar("loss/train", value["train_loss"], key)
            self.writer.add_scalar("acc/train", value["train_acc"], key)
        for key, value in self.valid_indices.items():
            self.writer.add_scalar("loss/valid", value["valid_loss"], key)
            self.writer.add_scalar("acc/valid", value["valid_acc"], key)

    @staticmethod
    def split(data_path_and_label_list, train_ratio=0.9):
        length = len(data_path_and_label_list)
        offset = int(length * train_ratio)
        shuffled_data_path_and_label_list = random.shuffle(data_path_and_label_list)
        return (
            shuffled_data_path_and_label_list[:offset],
            shuffled_data_path_and_label_list[offset:],
        )

    @staticmethod
    def get_acc(out, label):
        total = out.shape[0]
        _, pred_label = out.max(1)
        num_correct = (pred_label == label).sum().item()
        return num_correct / total

    def train(self) -> None:
        if self.device == "cuda":
            self.model = self.model.cuda()
            self.cls_loss = self.cls_loss.cuda()
        if self.resume:
            model_list = glob(os.path.join(self.result_path, "*.pt"))
            if len(model_list) != 0:
                model_list.sort()
                start_step = int(model_list[-1].split("_")[-1].split(".")[0])
                self.load_model(self.result_path, start_step)
                print("load success!")
        print("training starts!")

        start_time = time.time()
        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            # train
            self.model.train()
            print("train:")
            for x, label in tqdm(self.train_iter, total=len(self.train_iter)):
                self.forward_step += 1
                self.optim.zero_grad()
                x, label = (
                    x.to(dtype=torch.float),
                    label.to(dtype=torch.long),
                )
                if self.device == "cuda":
                    x, label = (x.cuda(), label.cuda())
                out = self.model(x)
                loss = self.cls_loss(out, label)
                loss.backward()
                self.optim.step()
                train_loss = loss.item()
                train_acc = self.get_acc(out, label)

                self.train_indices[self.forward_step] = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                }
                self.writer.add_scalar("loss/train", train_loss, self.forward_step)
                self.writer.add_scalar("acc/train", train_acc, self.forward_step)

            # valid
            self.model.eval()
            with torch.no_grad():
                for x, label in tqdm(self.valid_iter, total=len(self.valid_iter)):
                    x, label = (
                        x.to(dtype=torch.float),
                        label.to(dtype=torch.long),
                    )
                if self.device == "cuda":
                    x, label = (x.cuda(), label.cuda())
                out = self.model(x)
                loss = self.cls_loss(out, label)
                valid_loss = loss.item()
                valid_acc = self.get_acc(out, label)

                self.valid_indices[self.forward_step] = {
                    "valid_loss": valid_loss,
                    "valid_acc": valid_acc,
                }
                self.writer.add_scalar("loss/valid", valid_loss, self.forward_step)
                self.writer.add_scalar("acc/valid", valid_acc, self.forward_step)

            print(
                "Epoch %d time: %4.4f. \
                    Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f"
                % (
                    epoch,
                    time.time() - start_time,
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                )
            )

            if epoch % 1 == 0:
                self.start_epoch = epoch
                self.save_model(os.path.join(self.result_path, "model"), epoch)
