import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_curve,
    precision_recall_curve,
    auc,
)


class Model:
    """Model class predicts future price, up or down. Make scores of classification model's validation.
    Use LightGBM as a predictor model.
    Args:
        datamart: the stock datamart you want to predict
        feature: explanatory variables
    Example usage:
        if you want to predict up or down,
        model = Model(datamart, feature).predict()
        model[-1] is index of future prediction

        if you want to get scores of classification model,
        scores = Model(datamart, feature).score()
        you can get accuracy, precision, recall, f1, log loss, ROC-AUC, and PR-AUC
    """

    def __init__(self, datamart, feature):
        self._datamart = datamart
        self._feature = feature

    def avg_num(self, scores_list: list) -> list:
        """Get avarage scores in 5 times tests.
        Args:
            score_list: results of 5 time tests
        """
        self.avg = []
        for x in range(len(scores_list[0])):
            sum = 0
            for i in range(5):
                sum += scores_list[i][x]
            self.mae_avg = sum / 5
            self.avg.append(self.mae_avg)
        return self.avg

    def classify(self):
        self.clf = lgb.LGBMClassifier()
        self.df = pd.concat([self._datamart, self._feature], axis=1)
        self.df = self.df.iloc[::-1]
        self.df.reset_index(inplace=True, drop=True)
        self.X = self.df.iloc[:, 1:]
        self.y = self.df["target"]

    def folds_split(self):
        self.classify()
        self.folds = TimeSeriesSplit(n_splits=5)
        self.pred_lgb = []
        for train_index, test_index in self.folds.split(self.X):
            self.X_train, self.X_test = (
                self.X.iloc[
                    train_index,
                ],
                self.X.iloc[
                    test_index,
                ],
            )
            self.y_train, self.y_test = self.y[train_index], self.y[test_index]
            self.clf.fit(self.X_train, self.y_train)
            self.y_pred = self.clf.predict(self.X_test)
            self.pred_lgb.append(self.y_pred)

    def predict(self) -> list:
        self.folds_split()
        self.y_pred = self.avg_num(self.pred_lgb)
        self.y_pred = [round(x) for x in self.y_pred]
        return self.y_pred

    def score(self) -> str:
        self.classify()
        self.folds = TimeSeriesSplit(n_splits=5)
        self.scores_lgb = []
        for train_index, test_index in self.folds.split(self.X):
            self.X_train, self.X_test = (
                self.X.iloc[
                    train_index,
                ],
                self.X.iloc[
                    test_index,
                ],
            )
            self.y_train, self.y_test = self.y[train_index], self.y[test_index]
            self.clf.fit(self.X_train, self.y_train)
            self.y_pred = self.clf.predict(self.X_test)

            self.accuracy = accuracy_score(y_true=self.y_test, y_pred=self.y_pred)
            self.precision = precision_score(y_true=self.y_test, y_pred=self.y_pred)
            self.recall = recall_score(y_true=self.y_test, y_pred=self.y_pred)
            self.f1 = f1_score(y_true=self.y_test, y_pred=self.y_pred)
            self.probs = self.clf.predict_proba(self.X_test)
            self.log_loss = log_loss(y_true=self.y_test, y_pred=self.probs)
            self.y_score = self.probs[:, 1]
            self.fpr, self.tpr, self.thresholds = roc_curve(
                y_true=self.y_test, y_score=self.y_score
            )
            self.roc_auc = auc(self.fpr, self.tpr)
            (
                self.pr_precision,
                self.pr_recall,
                self.pr_thresholds,
            ) = precision_recall_curve(y_true=self.y_test, probas_pred=self.y_score)
            self.pr_auc = auc(self.pr_recall, self.pr_precision)
            self.scores_lgb.append(
                [
                    self.accuracy,
                    self.precision,
                    self.recall,
                    self.f1,
                    self.log_loss,
                    self.roc_auc,
                    self.pr_auc,
                ]
            )
        self.avg_score = self.avg_num(self.scores_lgb)
        return f"accuracy:{self.avg_score[0]}, precision:{self.avg_score[1]}, recall:{self.avg_score[2]}, f1:{self.avg_score[3]}, log loss:{self.avg_score[4]}, ROC-AUC:{self.avg_score[5]}, PR-AUC:{self.avg_score[6]}"
