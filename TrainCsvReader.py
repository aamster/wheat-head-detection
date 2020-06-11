import json

import numpy as np
import pandas as pd

class TrainCsvReader:
    def __init__(self, dir_input):
        self.dir_input = dir_input

    def preprocess(self):
        train_df = pd.read_csv(f'{self.dir_input}/train.csv')

        bboxes = np.vstack(train_df['bbox'].apply(lambda bbox: np.array(json.loads(bbox))))

        x2, y2 = self._convert_wh_to_xy_format(boxes=bboxes)
        train_df['x1'] = bboxes[:, 0]
        train_df['y1'] = bboxes[:, 1]
        train_df['x2'] = x2
        train_df['y2'] = y2

        train_df.drop(columns=['bbox'], inplace=True)

        return train_df

    @staticmethod
    def train_test_split(df, test_frac=.3):
        image_ids = df['image_id'].unique()
        train_cnt = int(len(image_ids)*(1-test_frac))
        train_ids = image_ids[:train_cnt]
        valid_ids = image_ids[train_cnt:]

        train_df = df[df['image_id'].isin(train_ids)]
        valid_df = df[df['image_id'].isin(valid_ids)]

        return train_df, valid_df

    @staticmethod
    def _convert_wh_to_xy_format(boxes):
        x1 = boxes[:, 0]
        width = boxes[:, 2]
        y1 = boxes[:, 1]
        height = boxes[:, 3]

        x2 = x1 + width
        y2 = y1 + height

        return x2, y2
