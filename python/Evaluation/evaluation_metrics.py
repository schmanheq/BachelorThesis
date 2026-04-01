from torchmetrics.classification import MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassConfusionMatrix 
from sklearn.metrics import matthews_corrcoef
import torch
import numpy as np

def basic_evaluation_metric(preds, target, mask):
    try:
        preds = torch.from_numpy(preds).long()
    except:
        pass
    try:
        target = torch.from_numpy(target).long()
    except:
        pass
    try:
        mask = torch.from_numpy(mask).long()
    except:
        pass
    
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

def confusion_matrix(preds,target,mask):
    try:
        preds = torch.from_numpy(preds).long()
    except:
        pass
    try:
        target = torch.from_numpy(target).long()
    except:
        pass
    try:
        mask = torch.from_numpy(mask).long()

    except:
        pass
    mask = ~(mask.bool())
    clean_preds = (preds-1)[mask.bool()]
    clean_target = (target-1)[mask.bool()]
    metric = MulticlassConfusionMatrix(num_classes=3)
    res = metric(clean_preds, clean_target)
    return res

def matth_coeff(preds,target,mask):
    try:
        preds = torch.from_numpy(preds).long()
    except:
        pass
    try:
        target = torch.from_numpy(target).long()
    except:
        pass
    try:
        mask = torch.from_numpy(mask).long()
    except:
        pass
    
    mask = ~(mask.bool())
    clean_preds = (preds-1)[mask.bool()]
    clean_target = (target-1)[mask.bool()]
    res = matthews_corrcoef(clean_preds, clean_target)
    return res

def custom_evaluation_metric(preds, batch_size):
    if preds.ndim != 3:
        preds = preds.reshape((batch_size, 10000, 90))
    preds = np.asanyarray(preds)
    violations = preds[..., 1:] >= preds[..., :-1]
    rows_with_errors = np.any(violations, axis=(1, 2))
    count = np.sum(rows_with_errors)
    return count

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