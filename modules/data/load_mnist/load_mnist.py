import sys
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from base_dockex.BaseDockex import BaseDockex


class LoadMNIST(BaseDockex):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.train_decimal = self.params["train_decimal"]
        self.valid_decimal = self.params["valid_decimal"]
        self.test_decimal = self.params["test_decimal"]
        self.num_samples = self.params["num_samples"]
        self.standardize_normalize = self.params["standardize_normalize"]
        self.random_seed = self.params["random_seed"]

        self.img_rows = 28
        self.img_cols = 28

        self.random_state = None
        self.X_flat = None
        self.y_str = None

        self.X_flat_train = None
        self.y_str_train = None

        self.X_flat_valid = None
        self.y_str_valid = None

        self.X_flat_test = None
        self.y_str_test = None

    def set_random_seed(self):
        print("Setting random seed")
        self.random_state = check_random_state(self.random_seed)

    def get_data(self):
        print("Getting data")
        self.X_flat, self.y_str = fetch_openml("mnist_784", version=1, return_X_y=True)

        self.X_flat = self.X_flat.astype('float32')

        if self.num_samples is not None:
            print("Trimming dataset to num_samples")
            self.X_flat = self.X_flat[0:self.num_samples]
            self.y_str = self.y_str[0:self.num_samples]

    def shuffle_data(self):
        print("Shuffling data")
        permutation = self.random_state.permutation(self.X_flat.shape[0])
        self.X_flat = self.X_flat[permutation]
        self.y_str = self.y_str[permutation]

    def train_valid_test_split(self):
        print("Splitting train / valid / test")
        num_samples = self.X_flat.shape[0]
        train_size = int(np.floor(self.train_decimal * num_samples))
        valid_size = int(np.floor(self.valid_decimal * num_samples))
        test_size = int(np.floor(self.test_decimal * num_samples))

        self.X_flat_train, X_flat_valid_test, self.y_str_train, y_str_valid_test = train_test_split(
            self.X_flat, self.y_str, train_size=train_size, test_size=(valid_size + test_size)
        )

        self.X_flat_valid, self.X_flat_test, self.y_str_valid, self.y_str_test = train_test_split(
            X_flat_valid_test, y_str_valid_test, train_size=valid_size, test_size=test_size
        )

    def standardize_features(self):
        print("Standardizing features")
        transformer = StandardScaler()
        self.X_flat_train = transformer.fit_transform(self.X_flat_train)
        self.X_flat_valid = transformer.transform(self.X_flat_valid)
        self.X_flat_test = transformer.transform(self.X_flat_test)
        
    def normalize_features(self):
        print("Normalizing features")
        self.X_flat_train /= 255
        self.X_flat_valid /= 255
        self.X_flat_test /= 255

    def save_outputs(self):
        print("Saving outputs")
        np.save(self.output_pathnames["X_flat_train_npy"], self.X_flat_train)
        np.save(self.output_pathnames["X_flat_valid_npy"], self.X_flat_valid)
        np.save(self.output_pathnames["X_flat_test_npy"], self.X_flat_test)
        
        np.save(self.output_pathnames["X_img_train_npy"], self.X_flat_train.reshape((self.X_flat_train.shape[0], self.img_rows, self.img_cols, 1)))
        np.save(self.output_pathnames["X_img_valid_npy"], self.X_flat_valid.reshape((self.X_flat_valid.shape[0], self.img_rows, self.img_cols, 1)))
        np.save(self.output_pathnames["X_img_test_npy"], self.X_flat_test.reshape((self.X_flat_test.shape[0], self.img_rows, self.img_cols, 1)))
                
        np.save(self.output_pathnames["y_str_train_npy"], self.y_str_train)
        np.save(self.output_pathnames["y_str_valid_npy"], self.y_str_valid)
        np.save(self.output_pathnames["y_str_test_npy"], self.y_str_test)
        
        y_int_train = self.y_str_train.astype('int')
        y_int_valid = self.y_str_valid.astype('int')
        y_int_test = self.y_str_test.astype('int')
        
        np.save(self.output_pathnames["y_int_train_npy"], y_int_train)
        np.save(self.output_pathnames["y_int_valid_npy"], y_int_valid)
        np.save(self.output_pathnames["y_int_test_npy"], y_int_test)

        lb = preprocessing.LabelBinarizer()
        lb.fit(y_int_train)
        
        np.save(self.output_pathnames["y_categorical_train_npy"], lb.transform(y_int_train))
        np.save(self.output_pathnames["y_categorical_valid_npy"], lb.transform(y_int_valid))
        np.save(self.output_pathnames["y_categorical_test_npy"], lb.transform(y_int_test))

    def run(self):
        print("Running")

        self.set_random_seed()

        self.get_data()

        self.shuffle_data()

        self.train_valid_test_split()

        if self.standardize_normalize == "standardize":
            self.standardize_features()
            
        elif self.standardize_normalize == "normalize":
            self.normalize_features()

        self.save_outputs()

        print("Success")


if __name__ == "__main__":
    LoadMNIST(sys.argv).run()
