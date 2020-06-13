from pathlib import Path

import pandas as pd
from plotly import express as px


class Plotter:
    def __init__(self):
        p = Path('plots')

        Path.mkdir(p, exist_ok=True)

    @staticmethod
    def plot_training_loss(losses):
        losses = pd.DataFrame(losses)
        losses = (
            losses.unstack()
                .reset_index()[['level_0', 0]]
                .rename(columns={'level_0': 'epoch', 0: 'loss'})
                .reset_index()
                .rename(columns={'index': 'iter'})
        )
        fig = px.line(losses, x='iter', y='loss', color='epoch')
        fig.write_html('plots/losses.html')

    @staticmethod
    def plot_validation_precision(precisions):
        precisions = pd.DataFrame(precisions).rename(columns={0: 'mean precision'})
        precisions['epoch'] = precisions.index

        fig = px.line(precisions, x='epoch', y='mean precision')
        fig.write_html('plots/validation_precision.html')

    def plot_metrics(self, losses, val_precisions):
        self.plot_training_loss(losses=losses)
        self.plot_validation_precision(precisions=val_precisions)