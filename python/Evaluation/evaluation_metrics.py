from torchmetrics.classification import MulticlassRecall, MulticlassPrecision, MulticlassF1Score
import torch
import numpy as np

def basic_evaluation_metric(preds, target, mask):
    preds = torch.from_numpy(preds).long()
    target = torch.from_numpy(target).long()
    mask = torch.from_numpy(mask).long()

    
    clean_preds = preds[mask].long() - 1
    clean_target = target[mask].long() - 1

    precision_metric = MulticlassPrecision(num_classes=3, average='none')
    recall_metric = MulticlassRecall(num_classes=3, average='none')
    f1_metric = MulticlassF1Score(num_classes=3, average='none')

    return (
        recall_metric(clean_preds, clean_target),
        precision_metric(clean_preds, clean_target),
        f1_metric(clean_preds, clean_target)
    )

def custom_evaluation_metric(preds):
    counter = 0
    for i in range(10000): #set to 10000 as we have 10000 rows
        for j in range(1,90):
            if preds[i][j]<preds[i][j-1]:
                counter+=1
                
    return (900000-counter)/900000

def custom_evaluation_metric_strict(preds,target):
    counter = 0
    for i in range(10000):
        if (preds[i]==target[i]).sum()!=90:
            counter+=1
    return (10000-counter)/10000