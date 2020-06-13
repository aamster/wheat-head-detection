import numpy as np
from pytorch_lightning.callbacks import ProgressBar


class CustomProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()


    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        bar.disable = True
        return bar

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.disable = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar

    def on_batch_end(self, trainer, pl_module):
        super().on_batch_end(trainer, pl_module)
        pl_module.update_epoch_loss(total_train_batches=self.total_train_batches, train_batch_idx=self.train_batch_idx)
