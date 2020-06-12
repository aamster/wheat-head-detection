import sys

from pytorch_lightning.callbacks import ProgressBar


class CustomProgressBar(ProgressBar):

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
        super().on_batch_end(trainer, pl_module)  # don't forget this :)
        percent = (self.train_batch_idx / self.total_train_batches) * 100
        sys.stdout.flush()
        sys.stdout.write(f'{percent:.01f} percent complete \r')

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        print(f'\tval mean precision: {pl_module.val_mean_precision}')
