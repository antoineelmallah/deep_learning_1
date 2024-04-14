import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(device, model, dataloader, num_classes, class_names):
    model.eval()

    num_correct = num_instances = 0.0
    cls_correct = {cls: 0 for cls in class_names}
    cls_num_instances = {cls: 0 for cls in class_names}

    for (x, y) in dataloader:
        batch_size = x.shape[0]
        num_instances += batch_size

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            out = model(x).argmax(dim=1)

        for label, pred in zip(y, out):
            if label == pred:
                cls_correct[breeds[label]] += 1
            cls_num_instances[breeds[label]] += 1

        num_correct += (out == y).sum().float()

    acc = lambda correct, total: 100.0 * correct / total
    accuracy_overall = acc(num_correct, num_instances).item()

    accuracy_cls = {}
    for cls, n_correct in cls_correct.items():
        accuracy_cls[cls] = acc(n_correct, cls_num_instances[cls])

    return accuracy_overall, accuracy_cls
