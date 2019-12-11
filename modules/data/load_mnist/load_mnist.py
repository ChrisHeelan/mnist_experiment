import sys
import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from base_dockex.BaseDockex import BaseDockex


class LoadMNIST(BaseDockex):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.train_decimal = self.params["train_decimal"]
        self.valid_decimal = self.params["valid_decimal"]
        self.test_decimal = self.params["test_decimal"]
        self.num_samples = self.params["num_samples"]
        self.standardize = self.params["standardize"]
        self.random_seed = self.params["random_seed"]

        self.random_state = None
        self.X = None
        self.y = None

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        self.transformer = None

    def set_random_seed(self):
        print("Setting random seed")
        self.random_state = check_random_state(self.random_seed)

    def get_data(self):
        print("Getting data")
        self.X, self.y = fetch_openml("mnist_784", version=1, return_X_y=True)
        self.X = self.X.reshape((self.X.shape[0], -1))

        if self.num_samples is not None:
            print("Trimming dataset to num_samples")
            self.X = self.X[0 : self.num_samples]
            self.y = self.y[0 : self.num_samples]

    def shuffle_data(self):
        print("Shuffling data")
        permutation = self.random_state.permutation(self.X.shape[0])
        self.X = self.X[permutation]
        self.y = self.y[permutation]

    def train_valid_test_split(self):
        print("Splitting train / valid / test")
        num_samples = self.X.shape[0]
        train_size = int(np.floor(self.train_decimal * num_samples))
        valid_size = int(np.floor(self.valid_decimal * num_samples))
        test_size = int(np.floor(self.test_decimal * num_samples))

        self.X_train, X_valid_test, self.y_train, y_valid_test = train_test_split(
            self.X, self.y, train_size=train_size, test_size=(valid_size + test_size)
        )

        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
            X_valid_test, y_valid_test, train_size=valid_size, test_size=test_size
        )

    def standardize_features(self):
        print("Standardizing features")
        self.transformer = StandardScaler()
        self.X_train = self.transformer.fit_transform(self.X_train)
        self.X_valid = self.transformer.transform(self.X_valid)
        self.X_test = self.transformer.transform(self.X_test)

    def save_outputs(self):
        print("Saving outputs")
        np.save(self.output_pathnames["X_train_npy"], self.X_train)
        np.save(self.output_pathnames["y_train_npy"], self.y_train)
        np.save(self.output_pathnames["X_valid_npy"], self.X_valid)
        np.save(self.output_pathnames["y_valid_npy"], self.y_valid)
        np.save(self.output_pathnames["X_test_npy"], self.X_test)
        np.save(self.output_pathnames["y_test_npy"], self.y_test)

        if self.transformer is not None:
            with open(
                self.output_pathnames["transform_joblib"], "wb"
            ) as transform_file:
                joblib.dump(self.transformer, transform_file)

    def run(self):
        print("Running")

        self.set_random_seed()

        self.get_data()

        self.shuffle_data()

        self.train_valid_test_split()

        if self.standardize:
            self.standardize_features()

        self.save_outputs()

        print("Success")


if __name__ == "__main__":
    LoadMNIST(sys.argv).run()
