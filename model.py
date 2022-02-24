import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from typing import Dict


class Model:
    """Model class splits train and test data-sets. Also, decide which predictor is better, LightGBM or LogisticRegressor.
    Args:
        model_str: predictor name,
        l_X_train: list of explanatory variables train,
        l_X_test: list of explanatory variables test,
        l_y_train: list of objective variables train,
        l_y_test: list of objective variables test,
    Example usage:
        used in ModelSelection class
    """

    def __init__(
        self,
        model_str: str,
        l_X_train: list,
        l_X_test: list,
        l_y_train: list,
        l_y_test: list,
    ):
        self.model_str = model_str
        self.l_X_train = l_X_train
        self.l_X_test = l_X_test
        self.l_y_train = l_y_train
        self.l_y_test = l_y_test

    @property
    def clf(self):
        if self.model_str == "lgb":
            return lgb.LGBMClassifier()
        elif self.model_str == "lr":
            return LogisticRegression(max_iter=1500)
        else:
            raise Exception(f"想定していないモデル({self.model_str})が指定されています。")

    def main(self) -> Dict[str, list]:
        """Return predictor name, predicted value, predicted probability, and probability of up.
        lists:
            self.l_clf = predictor name
            self.l_pred = predicted value
            self.l_prob = predicted probability
            self.l_prob_posi = probability of up
        """
        self.l_clf = []
        self.l_pred = []
        self.l_prob = []
        self.l_prob_posi = []
        for X_train, X_test, y_train in zip(
            self.l_X_train, self.l_X_test, self.l_y_train
        ):
            clf = self.clf
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            y_prob_posi = y_prob[:, 1]
            self.l_clf.append(clf)
            self.l_pred.append(y_pred)
            self.l_prob.append(y_prob)
            self.l_prob_posi.append(y_prob_posi)
