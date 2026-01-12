from torchmetrics.classification import MulticlassRecall, MulticlassPrecision
import torch
def basic_evaluation_metric(preds,target):
    precision_metric = MulticlassPrecision(num_classes=3, average='none')
    precision = precision_metric(preds,target)
    print(precision)
    recall_metric = MulticlassRecall(num_classes=3, average='none')
    recall = recall_metric(preds,target)
    print(recall)
