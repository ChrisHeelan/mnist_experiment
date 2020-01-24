import sys
from sklearn.ensemble import RandomForestClassifier

from base_dockex.BaseJoblibModel import BaseJoblibModel


class SklearnRandomForest(BaseJoblibModel):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.kwargs = self.params["kwargs"]

    def instantiate_model(self):
        self.model = RandomForestClassifier(**self.kwargs)


if __name__ == "__main__":
    SklearnRandomForest(sys.argv).run()
