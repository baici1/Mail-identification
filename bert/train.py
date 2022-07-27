from collections import OrderedDict
from glob import glob
import json
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.cuda.amp as amp

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
        self.record_step = config["record_step"]
        self.device = config["device"]
        self.seed = config["seed"]
        self.resume = config["resume"]

        self.train_indices = OrderedDict()
        self.valid_indices = OrderedDict()
        self.start_epoch = 0
        self.forward_step = 0

    def tensorboard_init(self) -> None:
        log_path = os.path.join(self.result_path, "log")
        try:
            shutil.rmtree(log_path)
            print("The folder of tensorboard has been emptied. Init!")
        except:
            print("The folder of tensorboard does not exist. Init!")
        self.writer = SummaryWriter(log_path)

    def setup_seed(self) -> None:
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def dataload(self) -> None:
        data_path_and_label_list = get_data(self.data_dir)
        train_data_path_and_label_list, vaild_data_path_and_label_list = self.split(
            data_path_and_label_list, 0.9
        )
        self.train_iter = DataLoader(
            dataset=TrainDataset(
                *zip(*train_data_path_and_label_list),
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
                *zip(*vaild_data_path_and_label_list),
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
        self.scaler = amp.GradScaler()

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
        random.shuffle(data_path_and_label_list)
        return (
            data_path_and_label_list[:offset],
            data_path_and_label_list[offset:],
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
            total_train_loss = 0
            total_train_acc = 0
            with tqdm(total=len(self.train_iter), ncols=100) as _tqdm:
                _tqdm.set_description("epoch: {}/{}".format(epoch, self.num_epochs))
                for *x, label in self.train_iter:
                    self.forward_step += 1
                    self.optim.zero_grad()
                    label = label.to(dtype=torch.long)
                    if self.device == "cuda":
                        x = [t.cuda() for t in x]
                        label = label.cuda()

                        with amp.autocast():
                            out = self.model(x)
                            loss = self.cls_loss(out, label)
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        out = self.model(x)
                        loss = self.cls_loss(out, label)
                        loss.backward()
                        self.optim.step()

                    train_loss = loss.item()
                    train_acc = self.get_acc(out, label)
                    total_train_loss += train_loss
                    total_train_acc += train_acc

                    _tqdm.set_postfix(
                        loss="{:.4f}".format(train_loss), acc="{:.4f}".format(train_acc)
                    )
                    _tqdm.update(1)
                    time.sleep(0.01)

                    if self.forward_step % self.record_step == 0:
                        self.train_indices[self.forward_step] = {
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                        }
                        self.writer.add_scalar(
                            "loss/train", train_loss, self.forward_step
                        )
                        self.writer.add_scalar(
                            "acc/train", train_acc, self.forward_step
                        )

            # valid
            print("valid:")
            self.model.eval()
            with torch.no_grad():
                total_valid_loss = 0
                total_valid_acc = 0
                with tqdm(total=len(self.valid_iter), ncols=100) as _tqdm:
                    _tqdm.set_description("epoch: {}/{}".format(epoch, self.num_epochs))
                    for *x, label in self.valid_iter:
                        label = label.to(dtype=torch.long)
                        if self.device == "cuda":
                            x = [t.cuda() for t in x]
                            label = label.cuda()
                        out = self.model(x)
                        loss = self.cls_loss(out, label)
                        valid_loss = loss.item()
                        valid_acc = self.get_acc(out, label)
                        total_valid_loss += valid_loss
                        total_valid_acc += valid_acc

                        _tqdm.set_postfix(
                            loss="{:.4f}".format(valid_loss),
                            acc="{:.4f}".format(valid_acc),
                        )
                        _tqdm.update(1)
                        time.sleep(0.01)

                self.valid_indices[epoch] = {
                    "valid_loss": total_valid_loss / len(self.valid_iter),
                    "valid_acc": total_valid_acc / len(self.valid_iter),
                }
                self.writer.add_scalar(
                    "loss/valid",
                    total_valid_loss / len(self.valid_iter),
                    epoch,
                )
                self.writer.add_scalar(
                    "acc/valid",
                    total_valid_acc / len(self.valid_iter),
                    epoch,
                )

            print(
                "Epoch %d time: %4.4f. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f"
                % (
                    epoch,
                    time.time() - start_time,
                    total_train_loss / len(self.train_iter),
                    total_train_acc / len(self.train_iter),
                    total_valid_loss / len(self.valid_iter),
                    total_valid_acc / len(self.valid_iter),
                )
            )

            if epoch % 1 == 0:
                self.start_epoch = epoch
                self.save_model(os.path.join(self.result_path, "model"), epoch)
        print("training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/train_config.json",
        help="the path of train config",
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
        print("config:")
        print(json.dumps(config, indent=4))
    train = Train(config=config)
    train.tensorboard_init()
    train.setup_seed()
    train.dataload()
    train.build_model()
    train.define_loss()
    train.define_optim()
    train.train()
