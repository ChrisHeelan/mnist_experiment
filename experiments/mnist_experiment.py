from dockex.core.experiment.ExperimentManager import ExperimentManager

NUM_SAMPLES = 20000

TRAIN_DECIMAL = 0.70
VALID_DECIMAL = 0.15
TEST_DECIMAL = 0.15

LOGISTIC_REGRESSION_PENALTY_LIST = ["l1", "l2", "none"]
LOGISTIC_REGRESSION_C_LIST = [1.0, 10.0, 50.0, 100.0]

RANDOM_SEED = 1337

if __name__ == "__main__":

    print("Running")
    manager = ExperimentManager(
        initial_job_num=0, experiment_name_prefix="mnist_experiment"
    )

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
            "standardize": True,
        },
        save_outputs=True,
        trial_tag="mnist_train_valid_test"
    )

    ######################################################################
    # scikit-learn logistic regression classification
    ######################################################################
    for logistic_regression_penalty in LOGISTIC_REGRESSION_PENALTY_LIST:
        for logistic_regression_C in LOGISTIC_REGRESSION_C_LIST:
            classifier_predictions = manager.add_job(
                "modules/classifiers/sklearn_logistic_regression",
                params={
                    "method": "fit_predict",
                    "kwargs": {
                        "C": logistic_regression_C,
                        "penalty": logistic_regression_penalty,
                        "solver": "saga",
                        "tol": 0.1,
                        "random_state": RANDOM_SEED,
                        "verbose": 1,
                    },
                    "divide_C_by_train_samples": True,
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
                skip_output_pathnames=["model_joblib"],
                save_outputs=True,
                trial_tag="classifier_predictions"
            )

            ######################################################################
            # generate classification report
            ######################################################################
            analysis = manager.add_job(
                "modules/analysis/sklearn_classification_report",
                input_pathnames={
                    "y_train_npy": mnist_train_valid_test["y_train_npy"],
                    "predict_train_npy": classifier_predictions["predict_train_npy"],
                    "y_valid_npy": mnist_train_valid_test["y_valid_npy"],
                    "predict_valid_npy": classifier_predictions["predict_valid_npy"],
                    "y_test_npy": mnist_train_valid_test["y_test_npy"],
                    "predict_test_npy": classifier_predictions["predict_test_npy"],
                },
                save_outputs=True,
                trial_tag="analysis",
                save_trial=True
            )

    ######################################################################
    # run the experiment
    ######################################################################
    manager.run()

    print("COMPLETE")
