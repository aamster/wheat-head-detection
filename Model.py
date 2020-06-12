from typing import Union, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn
import pytorch_lightning as pl

from ProgressBar import CustomProgressBar
from eval_metric import calculate_mean_precision


class WheatModule(pl.LightningModule):
    def __init__(self, model, lr=.005, momentum=0.9, weight_decay=0.0005):
        super().__init__()
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.val_mean_precision = None

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        lr_scheduler = None

        return [optimizer]

    def training_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # separate losses
        loss_dict = self.model(images, targets)

        # total loss
        loss = sum(loss for loss in loss_dict.values())

        logger_logs = {'training_loss': loss_dict}
        logger_logs = {'losses': logger_logs}

        output = {
            'loss': loss,
            # 'progress_bar': {'training_loss': loss},
            # 'log': logger_logs
        }

        return output

    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch

        preds = self.model(images)

        mean_precisions = torch.zeros(len(preds))
        mean_precisions = mean_precisions.type_as(images[0])
        thresholds = np.arange(.5, .8, .05)

        for i, pred in enumerate(preds):
            boxes_pred = pred['boxes']
            boxes_true = targets[i]['boxes']
            scores = pred['scores']
            mean_precisions[i] = calculate_mean_precision(boxes_pred=boxes_pred, boxes_true=boxes_true, scores=scores,
                                                      thresholds=thresholds)

        mean_mean_precision = mean_precisions.mean()
        logger_logs = {'validation_mean_precision': mean_mean_precision}

        output = {
            'validation_mean_precision': mean_mean_precision,
            # 'log': logger_logs
        }

        return output

    def validation_epoch_end(self, outputs):
        val_mean_precision = torch.stack([x['validation_mean_precision'] for x in outputs]).mean()
        self.val_mean_precision = val_mean_precision

        log = {'avg validation mean precision': val_mean_precision}

        return {
            'validation_mean_precision': val_mean_precision,
            'progress_bar': {'validation_mean_precision': val_mean_precision}
            # 'log': log
        }

    def predict(self, image_path):
        self.model.eval()
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device=DEVICE)
        outputs = self.model(image)

        outputs = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in outputs]
        boxes = outputs[0]["boxes"]
        scores = outputs[0]["scores"]

        # TODO should this logic be placed here and hardcoded at 0.5?
        valid_boxes = boxes[scores > 0.5]
        valid_scores = scores[scores > 0.5]

        return valid_boxes, valid_scores

    def get_prediction_string(self, test_images):
        res = []

        for image in test_images:
            boxes, scores = self.predict(image)
            prediction_string = []
            for (x_min, y_min, x_max, y_max), s in zip(boxes, scores):
                x = round(x_min)
                y = round(y_min)
                h = round(x_max - x_min)
                w = round(y_max - y_min)
                prediction_string.append(f"{s} {x} {y} {h} {w}")
            prediction_string = " ".join(prediction_string)

            res.append([image.replace('.jpg', ''), prediction_string])
        return pd.DataFrame(res, columns=["image_id", "PredictionString"])

    @staticmethod
    def _collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    from argparse import ArgumentParser
    from DataLoader import WheatDataLoader

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    # parser.add_argument('--batch_size', default=32)
    # parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('-dir_input')
    parser.add_argument('-dir_train')
    parser.add_argument('-num_workers', default=0, type=int)

    args = parser.parse_args()

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)

    dl = WheatDataLoader()
    train_dataloader, val_dataloader = dl.get_data_loaders(dir_input=args.dir_input, dir_train=args.dir_train,
                                                           debug=True, num_workers=args.num_workers)

    early_stop_callback = EarlyStopping(
        monitor='validation_mean_precision',
        min_delta=0.00,
        patience=1,
        verbose=True,
        mode='max'
    )

    progressBar = CustomProgressBar()
    trainer = pl.Trainer.from_argparse_args(args=args, early_stop_callback=early_stop_callback, callbacks=[progressBar])
    wheatModule = WheatModule(model=model)
    trainer.fit(model=wheatModule, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
