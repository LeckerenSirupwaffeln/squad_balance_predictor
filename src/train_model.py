# Module to train a tree-based model on Squad match-ups
# Learns to predict how balanced a Squad match-up is

# Python imports
from pathlib import Path
from typing import Optional
import statistics

# 3rd party imports
from loguru import logger
from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle
import pandas as pd
import lightgbm as lgb

# Relative imports
from .custom_types import (
    FEATURES_PREFIX,
    CATEGORICAL_FEATURES_PREFIX,
    NUMERICAL_TARGET_PREFIX,
    WEIGHT_COLUMN_NAME,
    K_FOLD_INDEX_COLUMN_NAME
)


def _load_data(file: Path) -> Optional[tuple[lgb.Dataset, pd.DataFrame]]:
    data = pd.read_csv(file)
    for column_name in data:
        if CATEGORICAL_FEATURES_PREFIX in column_name:
            data[column_name] = data[column_name].astype("category")
            logger.debug(f"{column_name} changed to categorical")

    x = data.filter(regex=FEATURES_PREFIX)
    y = data.filter(regex=NUMERICAL_TARGET_PREFIX)
    y_weights = data.filter(regex=WEIGHT_COLUMN_NAME)
    k_fold_indices = data.filter(regex=K_FOLD_INDEX_COLUMN_NAME)
    if x.empty or y.empty or y_weights.empty or k_fold_indices.empty:
        logger.error(f"Failed to load data from file: {file}")
        return None

    passed_filter = (
        len(y.columns) == 1 and
        len(y_weights.columns) == 1 and
        len(k_fold_indices.columns) == 1
    )
    if not passed_filter:
        logger.error(f"Loaded improper data from file: {file}")
        return None

    x, y, y_weights, k_fold_indices = shuffle(
        x, y, y_weights, k_fold_indices, random_state=13
    )
    dataset = lgb.Dataset(
        data=x,
        label=y.values,
        weight=y_weights.values,
        feature_name="auto",
        categorical_feature="auto",
        free_raw_data=False
    )
    dataset = dataset.construct()
    return dataset, k_fold_indices
# end _load_data() definition


def _generate_cv_folds(
        leftover_indices: list,
        test_idx: int,
        k_fold_indices: pd.DataFrame
        ) -> list[tuple[list[int], list[int]]]:
    output_list = []
    for validation_idx in leftover_indices:
        mask = k_fold_indices[K_FOLD_INDEX_COLUMN_NAME] == validation_idx
        validation_row_indices = k_fold_indices.index[mask]

        training_indices = [x for x in leftover_indices if x != validation_idx]
        mask = k_fold_indices[K_FOLD_INDEX_COLUMN_NAME].isin(training_indices)
        training_row_indices = k_fold_indices.index[mask]
        output_list.append((training_row_indices, validation_row_indices, ))

    return output_list
# end _generate_cv_folds() definition


def _nested_k_fold_cv(params: dict,
                      dataset: lgb.Dataset, k_fold_indices: pd.DataFrame) -> list:
    cross_val_scores = []
    unique_fold_indices = k_fold_indices[K_FOLD_INDEX_COLUMN_NAME].unique()
    for test_idx in unique_fold_indices:
        mask = k_fold_indices[K_FOLD_INDEX_COLUMN_NAME] == test_idx
        test_subset = dataset.subset(k_fold_indices.index[mask])

        leftover_indices = [x for x in unique_fold_indices if x != test_idx]
        cv_folds = _generate_cv_folds(
            leftover_indices,
            test_idx,
            k_fold_indices
        )

        results = lgb.cv(
            params=params,
            train_set=dataset,
            folds=cv_folds,
            shuffle=False,
            return_cvbooster=True
        )
        for booster in results["cvbooster"].boosters:
            test_eval = booster.eval(test_subset, "test")
            cross_val_scores.append(test_eval[0][2])

    return cross_val_scores
# end _nested_k_fold_cv() definition


def _hypertune(
        params: dict,
        training_dataset: lgb.Dataset,
        k_fold_indices: pd.DataFrame
    ) -> tuple[dict, float]:
    rmse_lowest_score = float("inf")
    rmse_lowest_score_params = None
    for params in list(ParameterGrid(params)):
        logger.info(f"Training on params: {params}")
        cross_val_scores = _nested_k_fold_cv(params, training_dataset, k_fold_indices)
        logger.info(f"Finished training on params: {params}")
        logger.info(f"Cross validation scores: {cross_val_scores}")
        mean_cross_val_scores = statistics.mean(cross_val_scores)
        logger.info(f"Mean cross validation scores: {mean_cross_val_scores}")
        if mean_cross_val_scores < rmse_lowest_score:
            rmse_lowest_score = mean_cross_val_scores
            rmse_lowest_score_params = params

    return rmse_lowest_score_params, rmse_lowest_score
# end _hypertune() definition


def _load_training_data(
    training_balance_file: Path,
    training_volatility_file: Path
    ) -> Optional[tuple[lgb.Dataset, pd.DataFrame, lgb.Dataset, pd.DataFrame]]:
    loaded_data = _load_data(training_balance_file)
    if loaded_data is None:
        logger.error(
            f"Failed to load training data from: {training_balance_file}"
        )
        return None

    balance_dataset, balance_k_fold_indices = loaded_data

    loaded_data = _load_data(training_volatility_file)
    if loaded_data is None:
        logger.error(
            f"Failed to load training data from: {training_volatility_file}"
        )
        return None

    volatility_dataset, volatility_k_fold_indices = loaded_data
    return (
        balance_dataset,
        balance_k_fold_indices,
        volatility_dataset,
        volatility_k_fold_indices
    )


@logger.catch
def train_model(
        training_balance_file: str,
        training_volatility_file: str,
        model_balance_file: str,
        model_volatility_file: str
    ) -> None:
    ABORT_MESSAGE = "Aborting train_model()"
    # Take care of file IO
    training_balance_file = Path(training_balance_file)
    if not training_balance_file.is_file():
        logger.error(
            f"training file: {training_balance_file} not found\n"
            f"{ABORT_MESSAGE}"
        )
        return

    training_volatility_file = Path(training_volatility_file)
    if not training_volatility_file.is_file():
        logger.error(
            f"training file: {training_volatility_file} not found\n"
            f"{ABORT_MESSAGE}"
        )
        return

    training_data = _load_training_data(training_balance_file, training_volatility_file)
    if training_data is None:
        logger.error(
            f"Failed to load training data\n"
            f"{ABORT_MESSAGE}"
        )
        return

    (
        balance_dataset,
        balance_k_fold_indices,
        volatility_dataset,
        volatility_k_fold_indices
    ) = training_data

    # Check if balance model file already exists
    model_balance_file = Path(model_balance_file)
    does_model_balance_file_exist = False
    if model_balance_file.is_file():
        logger.info(
            f"model file: {model_balance_file} already exists\n"
            f"Skipping balance model training"
        )
        does_model_balance_file_exist = True

    # Train balance model and export file if it doesn't exist already
    booster_balance = None
    if not does_model_balance_file_exist:
        logger.info("Training and cross-validating LGB balance model")
        params = {
            "metric": ["rmse"],
            "num_threads": [12],
            "boosting_type" : ["gbdt"],
            "min_data_in_leaf": [500],
            "feature_fraction": [0.66],
            "num_leaves": [20],
            "num_iterations": [1000],
            "early_stopping_round": [0],
            "learning_rate": [0.01],
            "verbose": [-1],
        }
        rmse_lowest_score_params, _ = _hypertune(
            params,
            balance_dataset,
            balance_k_fold_indices
        )

        booster_balance = lgb.train(
            params=rmse_lowest_score_params,
            train_set=balance_dataset,
            keep_training_booster=False
        )
        booster_balance.save_model(model_balance_file)
    else:
        booster_balance = lgb.Booster(model_file=model_balance_file)

    # Check if volatility model file already exists
    model_volatility_file = Path(model_volatility_file)
    does_model_volatility_file_exist = False
    if model_volatility_file.is_file():
        logger.info(
            f"model file: {model_volatility_file} already exists\n"
            f"Skipping volatility model training"
        )
        does_model_volatility_file_exist = True

    # Train volatility model and export file if it doesn't exist already
    if not does_model_volatility_file_exist:
        logger.info("Adjusting labels of training data to deviation squared")
        raw_data = volatility_dataset.get_data()
        raw_labels = volatility_dataset.get_label()
        if len(raw_data) != len(raw_labels):
            logger.error(
                "Raw data has length not equal to raw labels length\n"
                f"{ABORT_MESSAGE}"
            )
            return

        raw_data_predictions = booster_balance.predict(raw_data)
        for idx, _ in enumerate(raw_labels):
            y_actual = raw_labels[idx]
            y_pred = raw_data_predictions[idx]
            y_deviation = y_actual - y_pred
            y_deviation_squared = y_deviation ** 2
            raw_labels[idx] = y_deviation_squared

        volatility_dataset.set_label(raw_labels)
        volatility_dataset.construct()
        logger.info("Training and cross-validating LGB volatility model")
        params = {
            "metric": ["rmse"],
            "num_threads": [12],
            "boosting_type" : ["gbdt"],
            "min_data_in_leaf": [500],
            "feature_fraction": [0.66],
            "num_leaves": [20],
            "num_iterations": [1000],
            "early_stopping_round": [0],
            "learning_rate": [0.01],
            "verbose": [-1],
        }
        rmse_lowest_score_params, _ = _hypertune(
            params,
            volatility_dataset,
            volatility_k_fold_indices
        )
        booster_volatility = lgb.train(
            params=rmse_lowest_score_params,
            train_set=volatility_dataset,
            keep_training_booster=False
        )
        booster_volatility.save_model(model_volatility_file)
    return
# end train_model() definition
