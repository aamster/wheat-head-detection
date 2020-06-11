import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class WheatDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x1', 'y1', 'x2', 'y2']].values

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            image_aug = self.transforms(**{
                'image': image,
                'bboxes': target['boxes'],  # albumentations requires key of bboxes?
                'labels': labels
            })
            image = image_aug['image']
            target['boxes'] = image_aug['bboxes']
            # target['labels'] = image_aug['labels']

        target['boxes'] = torch.stack(tuple(map(torch.Tensor, zip(*target['boxes'])))).permute(1, 0)
        # target['boxes'] = torch.tensor(target['boxes'])
        target['boxes'] = target['boxes'].to(dtype=torch.float)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]