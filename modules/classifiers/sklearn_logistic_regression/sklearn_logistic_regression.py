import sys
from sklearn.linear_model import LogisticRegression

from base_dockex.BaseJoblibModel import BaseJoblibModel


class SklearnLogisticRegression(BaseJoblibModel):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.kwargs = self.params['kwargs']
        self.divide_C_by_train_samples = self.params['divide_C_by_train_samples']

    def instantiate_model(self):
        if self.divide_C_by_train_samples:
            if 'C' in self.kwargs:
                self.kwargs['C'] /= self.X_train.shape[0]
            else:
                self.kwargs['C'] = 1.0 / self.X_train.shape[0]

        self.model = LogisticRegression(**self.kwargs)


if __name__ == '__main__':
    SklearnLogisticRegression(sys.argv).run()
