import pandas as pd

from score import Score
from model import Model


class ModelSelection:
    """Decide which predictor is best."""

    def __init__(
        self,
        l_X_train: list,
        l_X_test: list,
        l_y_train: list,
        l_y_test: list,
        *model_str,
    ):
        self.model_str = list(model_str)
        self.l_X_train = l_X_train
        self.l_X_test = l_X_test
        self.l_y_train = l_y_train
        self.l_y_test = l_y_test

        self.l_score = []
        self.l_average_score = []
        self.l_ensemble_score = []
        self.l_l_clf = []

        self.n_split = 5

    def create_df_idx(self, score: Score):
        """Make pd.DataFrame with predictors."""
        v = [getattr(score, attr) for attr in dir(score) if attr.startswith("l_idx")]
        idx = [
            attr.replace("l_idx_", "")
            for attr in dir(score)
            if attr.startswith("l_idx")
        ]
        col = [order for order in range(1, self.n_split + 1)]
        return pd.DataFrame(v, index=idx, columns=col)

    def create_average_score(self, df_idx):
        """ある指標の中でどのモデルが良かったかの判断指標"""
        return df_idx.mean(axis=1)

    def create_ensemble_score(self, df_idx):
        """あるモデルの中で何番目の分割が良かったかの判断指標"""
        return df_idx.mean(axis=0)

    def _calc(self):
        """最良のモデル選択に使用する評価指標の数値を算出"""
        for model_str in self.model_str:
            model = Model(
                model_str, self.l_X_train, self.l_X_test, self.l_y_train, self.l_y_test
            )
            model.main()
            score = Score(model)
            df_idx = self.create_df_idx(score)
            average_score = self.create_average_score(df_idx)
            ensemble_score = self.create_ensemble_score(df_idx)
            self.l_average_score.append(average_score.sum())
            self.l_ensemble_score.append(ensemble_score.tolist())
            self.l_score.append(score)
            self.l_l_clf.append(model.l_clf)

    def best_model(self, verbose: bool = False):
        self._calc()
        max_ = max(self.l_average_score)
        self.idx_max_model = self.l_average_score.index(max_)

        max_ = max(self.l_ensemble_score[self.idx_max_model])
        self.idx_max_split = self.l_ensemble_score[self.idx_max_model].index(max_)

        if verbose:
            print(
                f"""best_model: {self.model_str[self.idx_max_model]}\nbest_split: {self.idx_max_split}"""
            )

        return self.l_l_clf[self.idx_max_model][self.idx_max_split]
