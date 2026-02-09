from torchmetrics.classification import MulticlassRecall, MulticlassPrecision, MulticlassF1Score
import torch
import numpy as np

def basic_evaluation_metric(preds, target, mask):
    if not torch.is_tensor(preds):
        preds = torch.from_numpy(preds).long()
        target = torch.from_numpy(target).long()
        mask = torch.from_numpy(mask).long()

    mask = ~(mask.bool())
    clean_preds = (preds-1)[mask.bool()]
    clean_target = (target-1)[mask.bool()]

    precision_metric = MulticlassPrecision(num_classes=3, average='none')
    recall_metric = MulticlassRecall(num_classes=3, average='none')
    f1_metric = MulticlassF1Score(num_classes=3, average='none')

    return (
        recall_metric(clean_preds, clean_target),
        precision_metric(clean_preds, clean_target),
        f1_metric(clean_preds, clean_target)
    )

def custom_evaluation_metric(preds, batch_size):
    if preds.ndim!=3:
        preds = preds.reshape((batch_size, 10000,90))
    preds = np.asanyarray(preds)
    violations = preds[..., 1:] < preds[..., :-1]
    return np.sum(np.sum(violations, axis=(1, 2))/900000)/batch_size

def custom_evaluation_metric_strict(preds,target, batch_size):
    if preds.ndim!=3:
        preds = preds.reshape((batch_size, 10000,90))
    if target.ndim!=3:
        target = target.reshape((batch_size, 10000,90))

    preds = np.asanyarray(preds)
    target = np.asanyarray(target)

    element_match = (preds == target)

    row_matches = np.all(element_match, axis=-1)
    correct_row_counts = np.sum(row_matches, axis=1)
    
    return np.sum(correct_row_counts / (10000))/batch_size