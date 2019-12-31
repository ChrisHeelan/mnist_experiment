import sys
from sklearn.neighbors import KNeighborsClassifier

from base_dockex.BaseJoblibModel import BaseJoblibModel


class SklearnKNN(BaseJoblibModel):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.kwargs = self.params["kwargs"]

    def instantiate_model(self):
        self.model = KNeighborsClassifier(**self.kwargs)


if __name__ == "__main__":
    SklearnKNN(sys.argv).run()
