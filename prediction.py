from model_selection import ModelSelection


class Prediction:
    """Apply bext model culculated by ModelSelection module.
    Example usage:
        pred = Prediction(model_selection, l_X_train, l_X_test, l_y_train, l_y_test).pred()
        if pred == 0, future preice would be down, 
        if pred == 1, future preice would be up 
    """

    def __init__(self, model_selection: ModelSelection, l_X_train: list, l_X_test: list, l_y_train: list, l_y_test: list):
        self.model_selection = model_selection
        self.l_X_train = l_X_train
        self.l_X_test = l_X_test
        self.l_y_train = l_y_train
        self.l_y_test = l_y_test
        
    def pred(self):
        self.best_model, self.idx_max_split = self.model_selection.best_model()
        self.best_model.fit(self.l_X_train[self.idx_max_split], self.l_y_train[self.idx_max_split])
        self.pred = self.best_model.predict(self.l_X_test[self.idx_max_split])
        return self.pred