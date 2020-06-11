from collections import defaultdict

import numpy as np
import torch
from torchvision.ops import box_iou


def _calculate_precision(boxes_true: torch.tensor, boxes_pred: torch.tensor, confidences: torch.tensor,
                         threshold=0.5) -> float:
    """Calculates precision for GT - prediction pairs at one threshold."""
    # edge case for no ground truth boxes
    if boxes_true.size(1) == 0:
        return 0.

    iou = box_iou(boxes1=boxes_pred, boxes2=boxes_true)

    pr_matches = set()
    gt_matches = set()

    # for each pred box, get list of ground truth boxes it matches with
    match_candidates = (iou >= threshold).nonzero()
    GT_PR_matches = defaultdict(list)
    for PR, GT in match_candidates:
        GT_PR_matches[GT.item()].append(PR.item())

    # Find which pred matches a GT box
    for GT, PRs in GT_PR_matches.items():
        # if multiple preds match a single ground truth box,
        # select the pred with the highest confidence
        if len(PRs) > 1:
            pr_match = PRs[confidences[PRs].argsort()[-1]]
        # else only a single pred matches this GT box
        else:
            pr_match = PRs[0]

        # only if we haven't seen a pred before can we mark a PR-GT pair as TP
        # otherwise the pred matches a different GT box and this GT might instead be a FN
        if pr_match not in pr_matches:
            gt_matches.add(GT)

        pr_matches.add(pr_match)

    TP = len(pr_matches)

    pr_idx = range(iou.size(0))
    gt_idx = range(iou.size(1))

    FP = len(set(pr_idx).difference(pr_matches))
    FN = len(set(gt_idx).difference(gt_matches))

    return TP / (TP + FP + FN)


def calculate_mean_precision(boxes_true: torch.tensor, boxes_pred: torch.tensor, scores: list, thresholds=(0.5,)) -> \
        float:
    """Calculates mean precision"""
    precision = np.zeros(len(thresholds))

    for i, threshold in enumerate(thresholds):
        precision[i] = _calculate_precision(boxes_true=boxes_true, boxes_pred=boxes_pred, confidences=scores,
                                                     threshold=threshold)
    return precision.mean()
