from dockex.core.experiment.ExperimentManager import ExperimentManager

ENABLE_GPUS = False

NUM_SAMPLES = 1000

TRAIN_DECIMAL = 0.70
VALID_DECIMAL = 0.15
TEST_DECIMAL = 0.15

STANDARDIZE_NORMALIZE_LIST = ["normalize"]

LOGISTIC_REGRESSION_PENALTY_LIST = ["l2"]
LOGISTIC_REGRESSION_C_LIST = [0.1, 1.0, 10.0]

KNN_N_NEIGHBORS_LIST = [4, 8, 16]
KNN_P_LIST = [1, 2]

RF_N_ESTIMATORS_LIST = [16, 32, 64]

CNN_FIRST_CNN_UNITS_LIST = [16]
CNN_SECOND_CNN_UNITS_LIST = [32]
CNN_DENSE_UNITS_LIST = [64]

RANDOM_SEED = 1337

if __name__ == "__main__":

    print("Running")
    manager = ExperimentManager(
        initial_job_num=0, experiment_name_prefix="mnist_experiment"
    )

    if ENABLE_GPUS:
        cnn_gpu_credits = 1

    else:
        cnn_gpu_credits = 0

    ######################################################################
    # generate train/valid/test MNIST dataset
    ######################################################################
    for standardize_normalize in STANDARDIZE_NORMALIZE_LIST:

        mnist_train_valid_test = manager.add_job(
            "modules/data/load_mnist",
            params={
                "train_decimal": TRAIN_DECIMAL,
                "valid_decimal": VALID_DECIMAL,
                "test_decimal": TEST_DECIMAL,
                "num_samples": NUM_SAMPLES,
                "standardize_normalize": standardize_normalize,
            },
            save_outputs=True,
            trial_tag="mnist_train_valid_test"
        )

        ######################################################################
        # scikit-learn logistic regression classifier
        ######################################################################
        for logistic_regression_penalty in LOGISTIC_REGRESSION_PENALTY_LIST:
            for logistic_regression_C in LOGISTIC_REGRESSION_C_LIST:

                classifier_predictions = manager.add_job(
                    "modules/classifiers/sklearn_logistic_regression",
                    params={
                        "method": "fit_then_predict",
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
                        "X_train_npy": mnist_train_valid_test["X_flat_train_npy"],
                        "y_train_npy": mnist_train_valid_test["y_str_train_npy"],
                        "X_valid_npy": mnist_train_valid_test["X_flat_valid_npy"],
                        "y_valid_npy": mnist_train_valid_test["y_str_valid_npy"],
                        "X_test_npy": mnist_train_valid_test["X_flat_test_npy"],
                        "y_test_npy": mnist_train_valid_test["y_str_test_npy"],
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
                        "y_train_npy": mnist_train_valid_test["y_str_train_npy"],
                        "predict_train_npy": classifier_predictions["predict_train_npy"],
                        "y_valid_npy": mnist_train_valid_test["y_str_valid_npy"],
                        "predict_valid_npy": classifier_predictions["predict_valid_npy"],
                        "y_test_npy": mnist_train_valid_test["y_str_test_npy"],
                        "predict_test_npy": classifier_predictions["predict_test_npy"],
                    },
                    save_outputs=True,
                    trial_tag="analysis",
                    save_trial=True
                )

        ######################################################################
        # scikit-learn KNN classifier
        ######################################################################
        for n_neighbors in KNN_N_NEIGHBORS_LIST:
            for p in KNN_P_LIST:

                classifier_predictions = manager.add_job(
                    "modules/classifiers/sklearn_knn",
                    params={
                        "method": "fit_then_predict",
                        "kwargs": {
                            "n_neighbors": n_neighbors,
                            "p": p,
                            "n_jobs": 1
                        }
                    },
                    input_pathnames={
                        "X_train_npy": mnist_train_valid_test["X_flat_train_npy"],
                        "y_train_npy": mnist_train_valid_test["y_str_train_npy"],
                        "X_valid_npy": mnist_train_valid_test["X_flat_valid_npy"],
                        "y_valid_npy": mnist_train_valid_test["y_str_valid_npy"],
                        "X_test_npy": mnist_train_valid_test["X_flat_test_npy"],
                        "y_test_npy": mnist_train_valid_test["y_str_test_npy"],
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
                        "y_train_npy": mnist_train_valid_test["y_str_train_npy"],
                        "predict_train_npy": classifier_predictions["predict_train_npy"],
                        "y_valid_npy": mnist_train_valid_test["y_str_valid_npy"],
                        "predict_valid_npy": classifier_predictions["predict_valid_npy"],
                        "y_test_npy": mnist_train_valid_test["y_str_test_npy"],
                        "predict_test_npy": classifier_predictions["predict_test_npy"],
                    },
                    save_outputs=True,
                    trial_tag="analysis",
                    save_trial=True
                )

        ######################################################################
        # scikit-learn Random Forest classifier
        ######################################################################
        for n_estimators in RF_N_ESTIMATORS_LIST:

            classifier_predictions = manager.add_job(
                "modules/classifiers/sklearn_random_forest",
                params={
                    "method": "fit_then_predict",
                    "kwargs": {
                        "n_estimators": n_estimators,
                        "n_jobs": 1,
                        "random_state": RANDOM_SEED,
                        "verbose": 1
                    }
                },
                input_pathnames={
                    "X_train_npy": mnist_train_valid_test["X_flat_train_npy"],
                    "y_train_npy": mnist_train_valid_test["y_str_train_npy"],
                    "X_valid_npy": mnist_train_valid_test["X_flat_valid_npy"],
                    "y_valid_npy": mnist_train_valid_test["y_str_valid_npy"],
                    "X_test_npy": mnist_train_valid_test["X_flat_test_npy"],
                    "y_test_npy": mnist_train_valid_test["y_str_test_npy"],
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
                    "y_train_npy": mnist_train_valid_test["y_str_train_npy"],
                    "predict_train_npy": classifier_predictions["predict_train_npy"],
                    "y_valid_npy": mnist_train_valid_test["y_str_valid_npy"],
                    "predict_valid_npy": classifier_predictions["predict_valid_npy"],
                    "y_test_npy": mnist_train_valid_test["y_str_test_npy"],
                    "predict_test_npy": classifier_predictions["predict_test_npy"],
                },
                save_outputs=True,
                trial_tag="analysis",
                save_trial=True
            )

        ######################################################################
        # Keras CNN classifier
        ######################################################################
        for first_cnn_units in CNN_FIRST_CNN_UNITS_LIST:
            for second_cnn_units in CNN_SECOND_CNN_UNITS_LIST:
                for dense_units in CNN_DENSE_UNITS_LIST:
                    classifier_predictions = manager.add_job(
                        "modules/classifiers/keras_cnn",
                        params={
                            "method": "fit_then_predict",
                            "first_cnn_units": first_cnn_units,
                            "second_cnn_units": second_cnn_units,
                            "dense_units": dense_units
                        },
                        input_pathnames={
                            "X_train_npy": mnist_train_valid_test["X_img_train_npy"],
                            "y_train_npy": mnist_train_valid_test["y_categorical_train_npy"],
                            "X_valid_npy": mnist_train_valid_test["X_img_valid_npy"],
                            "y_valid_npy": mnist_train_valid_test["y_categorical_valid_npy"],
                            "X_test_npy": mnist_train_valid_test["X_img_test_npy"],
                            "y_test_npy": mnist_train_valid_test["y_categorical_test_npy"],
                        },
                        gpu_credits=cnn_gpu_credits,
                        skip_output_pathnames=["model_keras"],
                        save_outputs=True,
                        trial_tag="classifier_predictions"
                    )

                    ######################################################################
                    # generate classification report
                    ######################################################################
                    analysis = manager.add_job(
                        "modules/analysis/sklearn_classification_report",
                        input_pathnames={
                            "y_train_npy": mnist_train_valid_test["y_categorical_train_npy"],
                            "predict_train_npy": classifier_predictions["predict_train_npy"],
                            "y_valid_npy": mnist_train_valid_test["y_categorical_valid_npy"],
                            "predict_valid_npy": classifier_predictions["predict_valid_npy"],
                            "y_test_npy": mnist_train_valid_test["y_categorical_test_npy"],
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
