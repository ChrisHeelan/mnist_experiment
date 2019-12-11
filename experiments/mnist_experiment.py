import copy
import pandas as pd

from dockex.core.experiment.ExperimentManager import ExperimentManager

NUM_SAMPLES = 20000

TRAIN_DECIMAL = 0.70
VALID_DECIMAL = 0.15
TEST_DECIMAL = 0.15

RANDOM_SEED = 1337

if __name__ == "__main__":

    print("Running")
    manager = ExperimentManager(
        initial_job_num=0,
        experiment_name_prefix='mnist_experiment'
    )

    run_dict_list = []
    run_dict = dict()

    ######################################################################
    # generate train/valid/test MNIST dataset
    ######################################################################
    mnist_train_valid_test = manager.add_job(
        "modules/data/load_mnist",
        params={
            "train_decimal": TRAIN_DECIMAL,
            "valid_decimal": VALID_DECIMAL,
            "test_decimal": TEST_DECIMAL,
            "num_samples": NUM_SAMPLES,
            "standardize": True
        },
        save_outputs=True
    )
    run_dict['mnist_train_valid_test'] = mnist_train_valid_test
    run_dict['mnist_train_valid_test_name'] = manager.job_list[-1]['name']

    ######################################################################
    # scikit-learn logistic regression classification
    ######################################################################
    classifier_predictions = manager.add_job(
        "modules/classifiers/sklearn_logistic_regression",
        params={
            "method": "fit_predict",
            "kwargs": {
                "C": 50.0,
                "penalty": "l1",
                "solver": "saga",
                "tol": 0.1,
                "random_state": RANDOM_SEED,
                "verbose": 1
            },
            "divide_C_by_train_samples": True
        },
        input_pathnames={
            "X_train_npy": mnist_train_valid_test["X_train_npy"],
            "y_train_npy": mnist_train_valid_test["y_train_npy"],
            "X_valid_npy": mnist_train_valid_test["X_valid_npy"],
            "y_valid_npy": mnist_train_valid_test["y_valid_npy"],
            "X_test_npy": mnist_train_valid_test["X_test_npy"],
            "y_test_npy": mnist_train_valid_test["y_test_npy"],
        },
        params_nested_update=True,
        skip_output_pathnames=['model_joblib'],
        save_outputs=True
    )
    run_dict['classifier_predictions'] = classifier_predictions
    run_dict['classifier_predictions_name'] = manager.job_list[-1]['name']

    ######################################################################
    # generate classification report
    ######################################################################
    analysis = manager.add_job(
        "modules/analysis/sklearn_classification_report",
        input_pathnames={
            "y_train_npy": mnist_train_valid_test['y_train_npy'],
            "predict_train_npy": classifier_predictions['predict_train_npy'],
            "y_valid_npy": mnist_train_valid_test['y_valid_npy'],
            "predict_valid_npy": classifier_predictions['predict_valid_npy'],
            "y_test_npy": mnist_train_valid_test['y_test_npy'],
            "predict_test_npy": classifier_predictions['predict_test_npy'],
        },
        save_outputs=True
    )
    run_dict['analysis'] = analysis
    run_dict['analysis_name'] = manager.job_list[-1]['name']

    run_dict_list.append(copy.deepcopy(run_dict))

    run_csv_filename = f"run_{manager.experiment_name}.csv"
    pd.DataFrame(run_dict_list).to_csv(f"/tmp/dockex/data/{run_csv_filename}")
    manager.send_to_output_saver(run_csv_filename)

    ######################################################################
    # run the experiment
    ######################################################################
    manager.run()

    print("COMPLETE")
