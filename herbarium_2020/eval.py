# --- 100 characters ------------------------------------------------------------------------------
# Created by: Shaun 2019/01/01

""" Evaluation"""
from pathlib import Path
import time

import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models

from tqdm import tqdm_notebook

import metrics
import model
import datatset
import utils


class Eval:
    def __init__(self, metric_man, loss, data, model, checkpoint_path):
        """Evaluation runner

        Args:
            metric_man (MetricManager):
            loss (torch.nn.modules.loss):
            data (DataLoader):
            model ():
            checkpoint_path (str): full path to checkpoint file if using `run once` or checkpoint
            dir if running during training.
        """

        self.metrics = metric_man
        self.loss = loss
        self.model = model
        self.data = data
        self.checkpoint_path = Path(checkpoint_path)

        self.step = 0
        self.current_check = -1

    def eval_step(self, batch):
        data, labels = batch
        preds = self.model(data)
        loss = self.loss(preds, labels)
        self.metrics.update(preds, labels, self.step)
        if self.metrics.writer:
            self.metrics.writer.add_scalar("loss", loss.item(), self.step)

    def run_once(self):
        """Used to run after epoch or just once on entire eval set"""
        self.load_checkpoint(self.checkpoint_path, gpu=True)
        for batch in tqdm_notebook(self.data):
            self.eval_step(batch)
            self.step += 1
        self.metrics.reset()

    def run(self):
        """Run eval continually while model is training"""
        while True:
            self.wait_load_new_checkpoint()
            for batch in tqdm_notebook(self.data):
                self.eval_step(batch)
                self.step += 1
            self.metrics.reset()

    def load_checkpoint(self, path, gpu=False):
        """Load checkpoint from directory"""
        print(f'Loading checkpoint {path}')
        self.model.load_state_dict(torch.load(path))
        if gpu:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.eval()

    def wait_load_new_checkpoint(self):
        """Waits till new checkpoint then loads checkpoint"""
        while True:
            models_idx = [int(m.stem.split("-")[-1]) for m in self.checkpoint_path.iterdir()]
            latest = np.max(models_idx) if len(models_idx) > 0 else -1
            if latest > self.current_check:
                self.current_check = latest
                self.load_checkpoint(f'{self.checkpoint_path}/model-{latest}.pth')
                break
            time.sleep(5)  # waiting 30 min


def run_eval(data_dir, checkpoints_dir, log_dir):

    # load and update labels
    df = pd.read_csv(data_dir / "train.csv")
    df = df[["file_name", "category_id"]]
    classes_map = utils.load_json(data_dir / "classes_map.json")

    # Update class labels
    new_labels = [classes_map[str(i)] for i in df["category_id"]]
    df["new_labels"] = new_labels

    ds = datatset.HerbDataSet(df, data_dir, 256, label="new_labels")
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    # Metrics
    acc = metrics.Accuracy()
    f1 = metrics.F1()
    writer = SummaryWriter(log_dir/"validation")
    manager = metrics.MetricManager([acc, f1], writer)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Model
    m = model.get_model(models.resnet50(pretrained=False), p1=0, p2=0,
                        device="gpu")  # no drop out, trying to over fit

    Eval(manager, criterion, loader, m, checkpoints_dir).run()
