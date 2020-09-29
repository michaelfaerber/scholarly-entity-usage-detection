import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def acc_and_f1(preds, labels):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    print(confusion_matrix(labels, preds))
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "precision": precision,
        "recall": recall
    }


def eval_end(outputs) -> tuple:
    """
    Calculate the mean loss, precision, recall, accuracy and f1-score.
    :param outputs: the result of validation_step()
    :return: the evaluation result, preds_list and labels_list
    """
    val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu()
    preds = np.concatenate([x["pred"] for x in outputs], axis=0)
    preds = np.argmax(preds, axis=1)

    out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    print(out_label_ids, preds)

    results = {**{"val_loss": val_loss_mean}, **acc_and_f1(preds, out_label_ids)}

    ret = {k: v for k, v in results.items()}
    ret["log"] = results
    return ret, preds_list, out_label_list
