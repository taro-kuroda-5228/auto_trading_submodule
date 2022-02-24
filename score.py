from model import Model

from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


class Score:
    """Return scores of prediction criteria, including accuracy_score, recall_score, precision_score, f1_score, pr-auc, and roc-auc."""

    def __init__(
        self,
        model: Model,
    ):
        self.model = model

    @property
    def l_idx_accuracy(self):
        l = []
        for true, pred in zip(self.model.l_y_test, self.model.l_pred):
            l.append(accuracy_score(y_true=true, y_pred=pred))
        return l

    @property
    def l_idx_precision(self):
        l = []
        for true, pred in zip(self.model.l_y_test, self.model.l_pred):
            l.append(precision_score(y_true=true, y_pred=pred))
        return l

    @property
    def l_idx_recall(self):
        l = []
        for true, pred in zip(self.model.l_y_test, self.model.l_pred):
            l.append(recall_score(y_true=true, y_pred=pred))
        return l

    @property
    def l_idx_f1(self):
        l = []
        for true, pred in zip(self.model.l_y_test, self.model.l_pred):
            l.append(f1_score(y_true=true, y_pred=pred))
        return l

    @property
    def l_idx_log_loss(self):
        l = []
        for true, prob in zip(self.model.l_y_test, self.model.l_prob):
            l.append(-1 * log_loss(y_true=true, y_pred=prob))
        return l

    @property
    def l_idx_roc_auc(self):
        l = []
        for true, prob_posi in zip(self.model.l_y_test, self.model.l_prob_posi):
            fpr, tpr, _ = roc_curve(y_true=true, y_score=prob_posi)
            l.append(auc(fpr, tpr))
        return l

    @property
    def l_idx_pr_auc(self):
        l = []
        for true, prob_posi in zip(self.model.l_y_test, self.model.l_prob_posi):
            pr_precision, pr_recall, _ = precision_recall_curve(
                y_true=true, probas_pred=prob_posi
            )
            l.append(auc(pr_recall, pr_precision))
        return l
