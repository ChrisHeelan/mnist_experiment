import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from base_dockex.BaseDockex import BaseDockex


class SklearnClassificationReport(BaseDockex):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.y_train = None
        self.predict_train = None
        self.y_valid = None
        self.predict_valid = None
        self.y_test = None
        self.predict_test = None

        self.report_dict_train = None
        self.report_dict_valid = None
        self.report_dict_test = None

    def load_input_arrays(self):
        print("Loading inputs")

        self.y_train = np.load(self.input_pathnames["y_train_npy"], allow_pickle=True)
        self.predict_train = np.load(
            self.input_pathnames["predict_train_npy"], allow_pickle=True
        )
        self.y_valid = np.load(self.input_pathnames["y_valid_npy"], allow_pickle=True)
        self.predict_valid = np.load(
            self.input_pathnames["predict_valid_npy"], allow_pickle=True
        )
        self.y_test = np.load(self.input_pathnames["y_test_npy"], allow_pickle=True)
        self.predict_test = np.load(
            self.input_pathnames["predict_test_npy"], allow_pickle=True
        )

        # if provided as categorical, take argmax
        if len(self.y_train.shape):
            self.y_train = np.argmax(self.y_train, axis=1)
            
        if len(self.predict_train.shape):
            self.predict_train = np.argmax(self.predict_train, axis=1)

        if len(self.y_valid.shape):
            self.y_valid = np.argmax(self.y_valid, axis=1)

        if len(self.predict_valid.shape):
            self.predict_valid = np.argmax(self.predict_valid, axis=1)

        if len(self.y_test.shape):
            self.y_test = np.argmax(self.y_test, axis=1)

        if len(self.predict_test.shape):
            self.predict_test = np.argmax(self.predict_test, axis=1)

    def generate_reports(self):
        print("Generating reports")

        self.report_dict_train = self.generate_classification_report(
            self.y_train, self.predict_train
        )
        self.report_dict_valid = self.generate_classification_report(
            self.y_valid, self.predict_valid
        )
        self.report_dict_test = self.generate_classification_report(
            self.y_test, self.predict_test
        )

    @staticmethod
    def generate_classification_report(y_true, y_pred):
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        return report_dict

    def save_outputs(self):
        pd.DataFrame([self.report_dict_train]).to_csv(
            self.output_pathnames["train_csv"]
        )
        pd.DataFrame([self.report_dict_valid]).to_csv(
            self.output_pathnames["valid_csv"]
        )
        pd.DataFrame([self.report_dict_test]).to_csv(self.output_pathnames["test_csv"])

    def run(self):
        print("Running")

        self.load_input_arrays()

        self.generate_reports()

        self.save_outputs()

        print("Success")


if __name__ == "__main__":
    SklearnClassificationReport(sys.argv).run()
