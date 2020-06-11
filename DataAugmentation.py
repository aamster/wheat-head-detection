import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(type='train'):
    transforms = []

    if type == 'train':
        transforms.append(A.Flip(0.5))

    transforms.append(ToTensorV2(p=1.0))

    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
