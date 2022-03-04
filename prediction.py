from model_selection import ModelSelection
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd


class Prediction:
    """Apply bext model culculated by ModelSelection module.
    Example usage:
        pred = Prediction(model_selection, l_X_train, l_X_test, l_y_train, l_y_test).pred()
        if pred == 0, future preice would be down,
        if pred == 1, future preice would be up
    """

    def __init__(
        self,
        model_selection: ModelSelection,
        l_X_train: list,
        l_X_test: list,
        l_y_train: list,
        l_y_test: list,
    ):
        self.model_selection = model_selection
        self.l_X_train = l_X_train
        self.l_X_test = l_X_test
        self.l_y_train = l_y_train
        self.l_y_test = l_y_test
        
    def fit(self):
        clf = lgb.LGBMClassifier()
        self.best_model, self.idx_max_split = clf, 4
        self.best_model.fit(self.l_X_train[self.idx_max_split], self.l_y_train[self.idx_max_split])
        
    def plot(self):
        self.fit()
        self.booster = self.best_model.booster_
        lgb.plot_importance(self.booster, importance_type="gain")
#         plt.show()
        
    def importance(self):
        self.plot()
        self.feat_imp = pd.DataFrame(
            {
                "特徴量": self.booster.feature_name(),
                "変数重要度": self.booster.feature_importance(importance_type="gain"),
            }
        )
        self.feat_imp["変数重要度"] = self.feat_imp["変数重要度"] / self.feat_imp["変数重要度"].sum()
        self.tmp = (
            self.feat_imp.sort_values(by="変数重要度", ascending=False)
            .reset_index(drop=True)
            .reset_index()
        )
        self.tmp["index"] = self.tmp["index"] + 1
        self.tmp.rename(columns={"index": "損失関数の減少に貢献したランキング"})
        return self.tmp.head()

    def pred(self):
        self.fit()
        self.pred = self.best_model.predict(self.l_X_test[self.idx_max_split])
        return self.pred
    
    def accuracy_score(self):
        self.pred()        
        self.accuracy_score = accuracy_score(y_true=self.l_y_test[self.idx_max_split], y_pred=self.pred)
        return self.accuracy_score
