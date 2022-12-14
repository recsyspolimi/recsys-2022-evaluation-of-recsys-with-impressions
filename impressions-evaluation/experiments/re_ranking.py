import os
import uuid
from typing import Optional, Sequence

import Recommenders.Recommender_import_list as recommenders
import attrs
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from recsys_framework_extensions.dask import DaskInterface
from recsys_framework_extensions.logging import get_logger

import experiments.commons as commons
from experiments.baselines import load_trained_recommender, TrainedRecommenderType
from impression_recommenders.re_ranking.impressions_discounting import DICT_SEARCH_CONFIGS as \
    ImpressionsDiscountingSearchConfigs
from impression_recommenders.user_profile.folding import FoldedMatrixFactorizationRecommender

logger = get_logger(__name__)


####################################################################################################
####################################################################################################
#                                FOLDERS VARIABLES                            #
####################################################################################################
####################################################################################################
BASE_FOLDER = os.path.join(
    commons.RESULTS_EXPERIMENTS_DIR,
    "re_ranking",
    "{benchmark}",
    "{evaluation_strategy}",
    "",
)
HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR = os.path.join(
    BASE_FOLDER,
    "experiments",
    ""
)

commons.FOLDERS.add(BASE_FOLDER)
commons.FOLDERS.add(HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR)


####################################################################################################
####################################################################################################
#                    Hyper-parameter tuning of Re-Ranking Recommender                              #
####################################################################################################
####################################################################################################
def _run_signal_analysis_ablation_impressions_re_ranking_hyper_parameter_tuning(
    experiment_case_ablation_reranking: commons.ExperimentCase,
    experiment_case_baseline: commons.ExperimentCase,
    experiment_baseline_similarity: Optional[str],
    try_folded_recommender: bool,
    signal_analysis_type: commons.SignalAnalysisType,
) -> None:
    """
    Runs in a dask worker the hyper-parameter tuning of a re-ranking impression recommender for the ablation study
    with positive or negative signals, as indicated by the `signal_analysis_type` variable.

    This method should not be called from outside.
    """
    experiment_can_be_executed = (
        experiment_case_ablation_reranking.benchmark == experiment_case_baseline.benchmark
        and experiment_case_ablation_reranking.hyper_parameter_tuning_parameters == experiment_case_baseline.hyper_parameter_tuning_parameters
    )

    if not experiment_can_be_executed:
        logger.warning(
            f"Early-returning from {_run_ablation_impressions_re_ranking_hyper_parameter_tuning.__name__} as it "
            f"received an invalid configuration. {experiment_case_ablation_reranking=} and {experiment_case_baseline=}"
        )
        return

    if commons.RecommenderImpressions.IMPRESSIONS_DISCOUNTING != experiment_case_ablation_reranking.recommender:
        logger.warning(
            f"Early-returning from {_run_ablation_impressions_re_ranking_hyper_parameter_tuning.__name__} as the "
            f"re-ranking impressions recommender for the ablation study is not "
            f"{commons.RecommenderImpressions.IMPRESSIONS_DISCOUNTING}. Received recommender "
            f"{experiment_case_ablation_reranking.recommender}"
        )
        return

    experiment_re_ranking_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_ablation_reranking.benchmark
    ]
    experiment_re_ranking_recommender = commons.MAPPER_ABLATION_AVAILABLE_RECOMMENDERS[
        experiment_case_ablation_reranking.recommender
    ]
    experiment_re_ranking_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_ablation_reranking.hyper_parameter_tuning_parameters
    ]

    experiment_baseline_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_baseline.benchmark
    ]
    experiment_baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case_baseline.recommender
    ]
    experiment_baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_baseline.hyper_parameter_tuning_parameters
    ]

    assert experiment_re_ranking_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_re_ranking_benchmark.config,
        benchmark=experiment_re_ranking_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy
    )

    impressions_feature_frequency_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_frequency_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_position_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_position_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_timestamp_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_timestamp_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    if commons.Benchmarks.FINNNoSlates == experiment_re_ranking_benchmark.benchmark:
        impressions_feature_last_seen_train = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
            )
        )
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    else:
        impressions_feature_last_seen_train = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
            )
        )
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    baseline_recommender_trained_train = load_trained_recommender(
        experiment_benchmark=experiment_baseline_benchmark,
        experiment_hyper_parameter_tuning_parameters=experiment_baseline_hyper_parameters,
        experiment_recommender=experiment_baseline_recommender,
        similarity=experiment_baseline_similarity,
        data_splits=interactions_data_splits,
        model_type=TrainedRecommenderType.TRAIN,
        try_folded_recommender=try_folded_recommender,
    )

    baseline_recommender_trained_train_validation = load_trained_recommender(
        experiment_benchmark=experiment_baseline_benchmark,
        experiment_hyper_parameter_tuning_parameters=experiment_baseline_hyper_parameters,
        experiment_recommender=experiment_baseline_recommender,
        similarity=experiment_baseline_similarity,
        data_splits=interactions_data_splits,
        model_type=TrainedRecommenderType.TRAIN_VALIDATION,
        try_folded_recommender=try_folded_recommender,
    )

    if baseline_recommender_trained_train is None or baseline_recommender_trained_train_validation is None:
        # We require a recommender that is already optimized.
        logger.warning(
            f"Early-skipping on {_run_impressions_re_ranking_hyper_parameter_tuning.__name__}. Could not load "
            f"trained recommenders for {experiment_baseline_recommender.recommender} with the benchmark "
            f"{experiment_baseline_benchmark.benchmark}. Folded Recommender? {try_folded_recommender}"
        )
        return

    instances_are_folded_recommenders = (
        isinstance(baseline_recommender_trained_train, FoldedMatrixFactorizationRecommender)
        and isinstance(baseline_recommender_trained_train_validation, FoldedMatrixFactorizationRecommender)
    )

    if try_folded_recommender and not instances_are_folded_recommenders:
        # Skip cases where the recommender cannot be folded.
        logger.warning(
            f"Skipping recommender {experiment_baseline_recommender.recommender} and "
            f"{experiment_re_ranking_recommender.recommender} because it cannot be folded and folded flag is set"
            f" to {True}"
        )
        return

    assert baseline_recommender_trained_train.RECOMMENDER_NAME == baseline_recommender_trained_train_validation.RECOMMENDER_NAME

    experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment_re_ranking_benchmark.benchmark.value,
        evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy.value,
    )
    experiment_file_name_prefix = f"{signal_analysis_type.value}_ABLATION_ONLY_UIM_FREQUENCY"
    experiment_file_name_root = (
        f"{experiment_file_name_prefix}"
        f"_{experiment_re_ranking_recommender.recommender.RECOMMENDER_NAME}"
        f"_{baseline_recommender_trained_train.RECOMMENDER_NAME}"
    )

    import random
    import numpy as np

    random.seed(experiment_re_ranking_hyper_parameters.reproducibility_seed)
    np.random.seed(experiment_re_ranking_hyper_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_re_ranking_hyper_parameters,
    )

    recommender_init_validation_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train.copy(),
            "uim_frequency": impressions_feature_frequency_train.copy(),
            "uim_position": impressions_feature_position_train.copy(),
            "uim_timestamp": impressions_feature_timestamp_train.copy(),
            "uim_last_seen": impressions_feature_last_seen_train.copy(),
            "seed": experiment_re_ranking_hyper_parameters.reproducibility_seed,
            "trained_recommender": baseline_recommender_trained_train,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    recommender_init_test_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train_validation.copy(),
            "uim_frequency": impressions_feature_frequency_train_validation.copy(),
            "uim_position": impressions_feature_position_train_validation.copy(),
            "uim_timestamp": impressions_feature_timestamp_train_validation.copy(),
            "uim_last_seen": impressions_feature_last_seen_train_validation.copy(),
            "seed": experiment_re_ranking_hyper_parameters.reproducibility_seed,
            "trained_recommender": baseline_recommender_trained_train_validation,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    hyper_parameter_search_space = attrs.asdict(
        ImpressionsDiscountingSearchConfigs[experiment_file_name_prefix],
    )

    logger_info = {
        "ablation_study": experiment_file_name_prefix,
        "re_ranking_recommender": experiment_re_ranking_recommender.recommender.RECOMMENDER_NAME,
        "baseline_recommender": experiment_baseline_recommender.recommender.RECOMMENDER_NAME,
        "dataset": experiment_re_ranking_benchmark.benchmark.value,
        "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
        "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
        "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": repr(experiment_re_ranking_hyper_parameters),
        "hyper_parameter_search_space": hyper_parameter_search_space,
    }

    logger.info(
        f"Hyper-parameter tuning arguments:"
        f"\n\t* {logger_info}"
    )

    search_bayesian_skopt = SearchBayesianSkopt(
        recommender_class=experiment_re_ranking_recommender.recommender,
        evaluator_validation=evaluators.validation,
        evaluator_test=evaluators.test,
        verbose=True,
    )
    search_bayesian_skopt.search(
        cutoff_to_optimize=experiment_re_ranking_hyper_parameters.cutoff_to_optimize,

        evaluate_on_test=experiment_re_ranking_hyper_parameters.evaluate_on_test,

        hyperparameter_search_space=hyper_parameter_search_space,

        max_total_time=experiment_re_ranking_hyper_parameters.max_total_time,
        metric_to_optimize=experiment_re_ranking_hyper_parameters.metric_to_optimize,

        n_cases=experiment_re_ranking_hyper_parameters.num_cases,
        n_random_starts=experiment_re_ranking_hyper_parameters.num_random_starts,

        output_file_name_root=experiment_file_name_root,
        output_folder_path=experiments_folder_path,

        recommender_input_args=recommender_init_validation_args_kwargs,
        recommender_input_args_last_test=recommender_init_test_args_kwargs,
        resume_from_saved=experiment_re_ranking_hyper_parameters.resume_from_saved,

        save_metadata=experiment_re_ranking_hyper_parameters.save_metadata,
        save_model=experiment_re_ranking_hyper_parameters.save_model,

        terminate_on_memory_error=experiment_re_ranking_hyper_parameters.terminate_on_memory_error,
    )


def _run_ablation_impressions_re_ranking_hyper_parameter_tuning(
    experiment_case_ablation_reranking: commons.ExperimentCase,
    experiment_case_baseline: commons.ExperimentCase,
    experiment_baseline_similarity: Optional[str],
    try_folded_recommender: bool,
) -> None:
    """
    Runs in a dask worker the hyper-parameter tuning of a re-ranking impression recommender for the ablation study.

    This method should not be called from outside.
    """
    experiment_can_be_executed = (
        experiment_case_ablation_reranking.benchmark == experiment_case_baseline.benchmark
        and experiment_case_ablation_reranking.hyper_parameter_tuning_parameters == experiment_case_baseline.hyper_parameter_tuning_parameters
    )

    if not experiment_can_be_executed:
        logger.warning(
            f"Early-returning from {_run_ablation_impressions_re_ranking_hyper_parameter_tuning.__name__} as it "
            f"received an invalid configuration. {experiment_case_ablation_reranking=} and {experiment_case_baseline=}"
        )
        return

    if commons.RecommenderImpressions.IMPRESSIONS_DISCOUNTING != experiment_case_ablation_reranking.recommender:
        logger.warning(
            f"Early-returning from {_run_ablation_impressions_re_ranking_hyper_parameter_tuning.__name__} as the "
            f"re-ranking impressions recommender for the ablation study is not "
            f"{commons.RecommenderImpressions.IMPRESSIONS_DISCOUNTING}. Received recommender "
            f"{experiment_case_ablation_reranking.recommender}"
        )
        return

    experiment_re_ranking_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_ablation_reranking.benchmark
    ]
    experiment_re_ranking_recommender = commons.MAPPER_ABLATION_AVAILABLE_RECOMMENDERS[
        experiment_case_ablation_reranking.recommender
    ]
    experiment_re_ranking_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_ablation_reranking.hyper_parameter_tuning_parameters
    ]

    experiment_baseline_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_baseline.benchmark
    ]
    experiment_baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case_baseline.recommender
    ]
    experiment_baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_baseline.hyper_parameter_tuning_parameters
    ]

    assert experiment_re_ranking_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_re_ranking_benchmark.config,
        benchmark=experiment_re_ranking_benchmark.benchmark,
    )

    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy
    )

    impressions_feature_frequency_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_frequency_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_position_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_position_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_timestamp_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_timestamp_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    if commons.Benchmarks.FINNNoSlates == experiment_re_ranking_benchmark.benchmark:
        impressions_feature_last_seen_train = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
            )
        )
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    else:
        impressions_feature_last_seen_train = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
            )
        )
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    baseline_recommender_trained_train = load_trained_recommender(
        experiment_benchmark=experiment_baseline_benchmark,
        experiment_hyper_parameter_tuning_parameters=experiment_baseline_hyper_parameters,
        experiment_recommender=experiment_baseline_recommender,
        similarity=experiment_baseline_similarity,
        data_splits=interactions_data_splits,
        model_type=TrainedRecommenderType.TRAIN,
        try_folded_recommender=try_folded_recommender,
    )

    baseline_recommender_trained_train_validation = load_trained_recommender(
        experiment_benchmark=experiment_baseline_benchmark,
        experiment_hyper_parameter_tuning_parameters=experiment_baseline_hyper_parameters,
        experiment_recommender=experiment_baseline_recommender,
        similarity=experiment_baseline_similarity,
        data_splits=interactions_data_splits,
        model_type=TrainedRecommenderType.TRAIN_VALIDATION,
        try_folded_recommender=try_folded_recommender,
    )

    if baseline_recommender_trained_train is None or baseline_recommender_trained_train_validation is None:
        # We require a recommender that is already optimized.
        logger.warning(
            f"Early-skipping on {_run_impressions_re_ranking_hyper_parameter_tuning.__name__}. Could not load "
            f"trained recommenders for {experiment_baseline_recommender.recommender} with the benchmark "
            f"{experiment_baseline_benchmark.benchmark}. Folded Recommender? {try_folded_recommender}"
        )
        return

    instances_are_folded_recommenders = (
        isinstance(baseline_recommender_trained_train, FoldedMatrixFactorizationRecommender)
        and isinstance(baseline_recommender_trained_train_validation, FoldedMatrixFactorizationRecommender)
    )

    if try_folded_recommender and not instances_are_folded_recommenders:
        # Skip cases where the recommender cannot be folded.
        logger.warning(
            f"Skipping recommender {experiment_baseline_recommender.recommender} and "
            f"{experiment_re_ranking_recommender.recommender} because it cannot be folded and folded flag is set"
            f" to {True}"
        )
        return

    assert baseline_recommender_trained_train.RECOMMENDER_NAME == baseline_recommender_trained_train_validation.RECOMMENDER_NAME

    experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
        benchmark=experiment_re_ranking_benchmark.benchmark.value,
        evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy.value,
    )
    experiment_file_name_prefix = "ABLATION_UIM_FREQUENCY"
    experiment_file_name_root = (
        f"{experiment_file_name_prefix}"
        f"_{experiment_re_ranking_recommender.recommender.RECOMMENDER_NAME}"
        f"_{baseline_recommender_trained_train.RECOMMENDER_NAME}"
    )

    import random
    import numpy as np

    random.seed(experiment_re_ranking_hyper_parameters.reproducibility_seed)
    np.random.seed(experiment_re_ranking_hyper_parameters.reproducibility_seed)

    evaluators = commons.get_evaluators(
        data_splits=interactions_data_splits,
        experiment_hyper_parameter_tuning_parameters=experiment_re_ranking_hyper_parameters,
    )

    recommender_init_validation_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train.copy(),
            "uim_frequency": impressions_feature_frequency_train.copy(),
            "uim_position": impressions_feature_position_train.copy(),
            "uim_timestamp": impressions_feature_timestamp_train.copy(),
            "uim_last_seen": impressions_feature_last_seen_train.copy(),
            "seed": experiment_re_ranking_hyper_parameters.reproducibility_seed,
            "trained_recommender": baseline_recommender_trained_train,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    recommender_init_test_args_kwargs = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[],
        CONSTRUCTOR_KEYWORD_ARGS={
            "urm_train": interactions_data_splits.sp_urm_train_validation.copy(),
            "uim_frequency": impressions_feature_frequency_train_validation.copy(),
            "uim_position": impressions_feature_position_train_validation.copy(),
            "uim_timestamp": impressions_feature_timestamp_train_validation.copy(),
            "uim_last_seen": impressions_feature_last_seen_train_validation.copy(),
            "seed": experiment_re_ranking_hyper_parameters.reproducibility_seed,
            "trained_recommender": baseline_recommender_trained_train_validation,
        },
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    hyper_parameter_search_space = attrs.asdict(
        ImpressionsDiscountingSearchConfigs["ABLATION_ONLY_UIM_FREQUENCY"],
    )

    logger_info = {
        "ablation_study": "ABLATION_ONLY_UIM_FREQUENCY",
        "re_ranking_recommender": experiment_re_ranking_recommender.recommender.RECOMMENDER_NAME,
        "baseline_recommender": experiment_baseline_recommender.recommender.RECOMMENDER_NAME,
        "dataset": experiment_re_ranking_benchmark.benchmark.value,
        "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
        "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
        "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
        "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
        "hyper_parameter_tuning_parameters": repr(experiment_re_ranking_hyper_parameters),
        "hyper_parameter_search_space": hyper_parameter_search_space,
    }

    logger.info(
        f"Hyper-parameter tuning arguments:"
        f"\n\t* {logger_info}"
    )

    search_bayesian_skopt = SearchBayesianSkopt(
        recommender_class=experiment_re_ranking_recommender.recommender,
        evaluator_validation=evaluators.validation,
        evaluator_test=evaluators.test,
        verbose=True,
    )
    search_bayesian_skopt.search(
        cutoff_to_optimize=experiment_re_ranking_hyper_parameters.cutoff_to_optimize,

        evaluate_on_test=experiment_re_ranking_hyper_parameters.evaluate_on_test,

        hyperparameter_search_space=hyper_parameter_search_space,

        max_total_time=experiment_re_ranking_hyper_parameters.max_total_time,
        metric_to_optimize=experiment_re_ranking_hyper_parameters.metric_to_optimize,

        n_cases=experiment_re_ranking_hyper_parameters.num_cases,
        n_random_starts=experiment_re_ranking_hyper_parameters.num_random_starts,

        output_file_name_root=experiment_file_name_root,
        output_folder_path=experiments_folder_path,

        recommender_input_args=recommender_init_validation_args_kwargs,
        recommender_input_args_last_test=recommender_init_test_args_kwargs,
        resume_from_saved=experiment_re_ranking_hyper_parameters.resume_from_saved,

        save_metadata=experiment_re_ranking_hyper_parameters.save_metadata,
        save_model=experiment_re_ranking_hyper_parameters.save_model,

        terminate_on_memory_error=experiment_re_ranking_hyper_parameters.terminate_on_memory_error,
    )


def _run_impressions_re_ranking_hyper_parameter_tuning(
    experiment_case_reranking: commons.ExperimentCase,
    experiment_case_baseline: commons.ExperimentCase,
    experiment_baseline_similarity: Optional[str],
) -> None:
    """
    Runs in a dask worker the hyper-parameter tuning of a re-ranking impression recommender.

    This method should not be called from outside.
    """

    experiment_re_ranking_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_reranking.benchmark
    ]
    experiment_re_ranking_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case_reranking.recommender
    ]
    experiment_re_ranking_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_reranking.hyper_parameter_tuning_parameters
    ]

    experiment_baseline_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
        experiment_case_baseline.benchmark
    ]
    experiment_baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
        experiment_case_baseline.recommender
    ]
    experiment_baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
        experiment_case_baseline.hyper_parameter_tuning_parameters
    ]

    assert experiment_re_ranking_recommender.search_hyper_parameters is not None

    benchmark_reader = commons.get_reader_from_benchmark(
        benchmark_config=experiment_re_ranking_benchmark.config,
        benchmark=experiment_re_ranking_benchmark.benchmark,
    )
    
    dataset = benchmark_reader.dataset

    interactions_data_splits = dataset.get_urm_splits(
        evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy
    )

    impressions_feature_frequency_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_frequency_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_FREQUENCY,
            impressions_feature_column=commons.ImpressionsFeatureColumnsFrequency.FREQUENCY,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_position_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_position_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_POSITION,
            impressions_feature_column=commons.ImpressionsFeatureColumnsPosition.POSITION,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    impressions_feature_timestamp_train = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
        )
    )
    impressions_feature_timestamp_train_validation = dataset.sparse_matrix_impression_feature(
        feature=commons.get_feature_key_by_benchmark(
            benchmark=experiment_re_ranking_benchmark.benchmark,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
            impressions_feature=commons.ImpressionsFeatures.USER_ITEM_TIMESTAMP,
            impressions_feature_column=commons.ImpressionsFeatureColumnsTimestamp.TIMESTAMP,
            impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
        )
    )

    if commons.Benchmarks.FINNNoSlates == experiment_re_ranking_benchmark.benchmark:
        impressions_feature_last_seen_train = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
            )
        )
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.EUCLIDEAN,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    else:
        impressions_feature_last_seen_train = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN,
            )
        )
        impressions_feature_last_seen_train_validation = dataset.sparse_matrix_impression_feature(
            feature=commons.get_feature_key_by_benchmark(
                benchmark=experiment_re_ranking_benchmark.benchmark,
                evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy,
                impressions_feature=commons.ImpressionsFeatures.USER_ITEM_LAST_SEEN,
                impressions_feature_column=commons.ImpressionsFeatureColumnsLastSeen.TOTAL_DAYS,
                impressions_feature_split=commons.ImpressionsFeaturesSplit.TRAIN_VALIDATION,
            )
        )

    for try_folded_recommender in [True, False]:
        baseline_recommender_trained_train = load_trained_recommender(
            experiment_benchmark=experiment_baseline_benchmark,
            experiment_hyper_parameter_tuning_parameters=experiment_baseline_hyper_parameters,
            experiment_recommender=experiment_baseline_recommender,
            similarity=experiment_baseline_similarity,
            data_splits=interactions_data_splits,
            model_type=TrainedRecommenderType.TRAIN,
            try_folded_recommender=try_folded_recommender,
        )

        baseline_recommender_trained_train_validation = load_trained_recommender(
            experiment_benchmark=experiment_baseline_benchmark,
            experiment_hyper_parameter_tuning_parameters=experiment_baseline_hyper_parameters,
            experiment_recommender=experiment_baseline_recommender,
            similarity=experiment_baseline_similarity,
            data_splits=interactions_data_splits,
            model_type=TrainedRecommenderType.TRAIN_VALIDATION,
            try_folded_recommender=try_folded_recommender,
        )

        if baseline_recommender_trained_train is None or baseline_recommender_trained_train_validation is None:
            # We require a recommender that is already optimized.
            logger.warning(
                f"Early-skipping on {_run_impressions_re_ranking_hyper_parameter_tuning.__name__}. Could not load "
                f"trained recommenders for {experiment_baseline_recommender.recommender} with the benchmark "
                f"{experiment_baseline_benchmark.benchmark}. Folded Recommender? {try_folded_recommender}"
            )
            continue

        instances_are_folded_recommenders = (
            isinstance(baseline_recommender_trained_train, FoldedMatrixFactorizationRecommender)
            and isinstance(baseline_recommender_trained_train_validation, FoldedMatrixFactorizationRecommender)
        )

        if try_folded_recommender and not instances_are_folded_recommenders:
            # Skip cases where the recommender cannot be folded.
            logger.warning(
                f"Skipping recommender {experiment_baseline_recommender.recommender} and "
                f"{experiment_re_ranking_recommender.recommender} because it cannot be folded and folded flag is set"
                f" to {True}"
            )
            continue

        assert baseline_recommender_trained_train.RECOMMENDER_NAME == baseline_recommender_trained_train_validation.RECOMMENDER_NAME

        experiments_folder_path = HYPER_PARAMETER_TUNING_EXPERIMENTS_DIR.format(
            benchmark=experiment_re_ranking_benchmark.benchmark.value,
            evaluation_strategy=experiment_re_ranking_hyper_parameters.evaluation_strategy.value,
        )
        experiment_file_name_root = (
            f"{experiment_re_ranking_recommender.recommender.RECOMMENDER_NAME}"
            f"_{baseline_recommender_trained_train.RECOMMENDER_NAME}"
        )

        import random
        import numpy as np

        random.seed(experiment_re_ranking_hyper_parameters.reproducibility_seed)
        np.random.seed(experiment_re_ranking_hyper_parameters.reproducibility_seed)

        evaluators = commons.get_evaluators(
            data_splits=interactions_data_splits,
            experiment_hyper_parameter_tuning_parameters=experiment_re_ranking_hyper_parameters,
        )

        recommender_init_validation_args_kwargs = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS=[],
            CONSTRUCTOR_KEYWORD_ARGS={
                "urm_train": interactions_data_splits.sp_urm_train.copy(),
                "uim_frequency": impressions_feature_frequency_train.copy(),
                "uim_position": impressions_feature_position_train.copy(),
                "uim_timestamp": impressions_feature_timestamp_train.copy(),
                "uim_last_seen": impressions_feature_last_seen_train.copy(),
                "seed": experiment_re_ranking_hyper_parameters.reproducibility_seed,
                "trained_recommender": baseline_recommender_trained_train,
            },
            FIT_POSITIONAL_ARGS=[],
            FIT_KEYWORD_ARGS={},
            EARLYSTOPPING_KEYWORD_ARGS={},
        )

        recommender_init_test_args_kwargs = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS=[],
            CONSTRUCTOR_KEYWORD_ARGS={
                "urm_train": interactions_data_splits.sp_urm_train_validation.copy(),
                "uim_frequency": impressions_feature_frequency_train_validation.copy(),
                "uim_position": impressions_feature_position_train_validation.copy(),
                "uim_timestamp": impressions_feature_timestamp_train_validation.copy(),
                "uim_last_seen": impressions_feature_last_seen_train_validation.copy(),
                "seed": experiment_re_ranking_hyper_parameters.reproducibility_seed,
                "trained_recommender": baseline_recommender_trained_train_validation,
            },
            FIT_POSITIONAL_ARGS=[],
            FIT_KEYWORD_ARGS={},
            EARLYSTOPPING_KEYWORD_ARGS={},
        )

        hyper_parameter_search_space = attrs.asdict(
            experiment_re_ranking_recommender.search_hyper_parameters()
        )

        logger_info = {
            "re_ranking_recommender": experiment_re_ranking_recommender.recommender.RECOMMENDER_NAME,
            "baseline_recommender": experiment_baseline_recommender.recommender.RECOMMENDER_NAME,
            "dataset": experiment_re_ranking_benchmark.benchmark.value,
            "urm_test_shape": interactions_data_splits.sp_urm_test.shape,
            "urm_train_shape": interactions_data_splits.sp_urm_train.shape,
            "urm_validation_shape": interactions_data_splits.sp_urm_validation.shape,
            "urm_train_and_validation_shape": interactions_data_splits.sp_urm_train_validation.shape,
            "hyper_parameter_tuning_parameters": repr(experiment_re_ranking_hyper_parameters),
            "hyper_parameter_search_space": hyper_parameter_search_space,
        }

        logger.info(
            f"Hyper-parameter tuning arguments:"
            f"\n\t* {logger_info}"
        )

        search_bayesian_skopt = SearchBayesianSkopt(
            recommender_class=experiment_re_ranking_recommender.recommender,
            evaluator_validation=evaluators.validation,
            evaluator_test=evaluators.test,
            verbose=True,
        )
        search_bayesian_skopt.search(
            cutoff_to_optimize=experiment_re_ranking_hyper_parameters.cutoff_to_optimize,

            evaluate_on_test=experiment_re_ranking_hyper_parameters.evaluate_on_test,

            hyperparameter_search_space=hyper_parameter_search_space,

            max_total_time=experiment_re_ranking_hyper_parameters.max_total_time,
            metric_to_optimize=experiment_re_ranking_hyper_parameters.metric_to_optimize,

            n_cases=experiment_re_ranking_hyper_parameters.num_cases,
            n_random_starts=experiment_re_ranking_hyper_parameters.num_random_starts,

            output_file_name_root=experiment_file_name_root,
            output_folder_path=experiments_folder_path,

            recommender_input_args=recommender_init_validation_args_kwargs,
            recommender_input_args_last_test=recommender_init_test_args_kwargs,
            resume_from_saved=experiment_re_ranking_hyper_parameters.resume_from_saved,

            save_metadata=experiment_re_ranking_hyper_parameters.save_metadata,
            save_model=experiment_re_ranking_hyper_parameters.save_model,

            terminate_on_memory_error=experiment_re_ranking_hyper_parameters.terminate_on_memory_error,
        )


def run_impressions_re_ranking_experiments(
    dask_interface: DaskInterface,
    re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    """
    Public method that instructs dask to run in dask workers the hyper-parameter tuning of the impressions discounting
    recommenders.

    Processes are always preferred than threads as the hyper-parameter tuning loop is probably not thread-safe.
    """
    for experiment_case_reranking in re_ranking_experiment_cases_interface.experiment_cases:
        for experiment_case_baseline in baseline_experiment_cases_interface.experiment_cases:
            if (
                experiment_case_reranking.benchmark != experiment_case_baseline.benchmark
                and experiment_case_reranking.hyper_parameter_tuning_parameters != experiment_case_baseline.hyper_parameter_tuning_parameters
            ):
                continue

            re_ranking_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
                experiment_case_reranking.benchmark
            ]
            re_ranking_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
                experiment_case_reranking.recommender
            ]

            baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
                experiment_case_baseline.recommender
            ]
            baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
                experiment_case_baseline.hyper_parameter_tuning_parameters
            ]

            similarities: Sequence[Optional[commons.T_SIMILARITY_TYPE]] = [None]
            if baseline_recommender.recommender in [
                recommenders.ItemKNNCFRecommender, recommenders.UserKNNCFRecommender
            ]:
                similarities = baseline_hyper_parameters.knn_similarity_types
            for similarity in similarities:
                dask_interface.submit_job(
                    job_key=(
                        f"_run_impressions_heuristics_hyper_parameter_tuning"
                        f"|{experiment_case_reranking.benchmark.value}"
                        f"|{re_ranking_recommender.recommender.RECOMMENDER_NAME}"
                        f"|{baseline_recommender.recommender.RECOMMENDER_NAME}"
                        f"|{similarity}"
                        f"|{uuid.uuid4()}"
                    ),
                    job_priority=(
                        re_ranking_benchmark.priority
                        * re_ranking_recommender.priority
                        * baseline_recommender.priority
                    ),
                    job_info={
                        "recommender": re_ranking_recommender.recommender.RECOMMENDER_NAME,
                        "baseline": baseline_recommender.recommender.RECOMMENDER_NAME,
                        "similarity": similarity,
                        "benchmark": re_ranking_benchmark.benchmark.value,
                    },
                    method=_run_impressions_re_ranking_hyper_parameter_tuning,
                    method_kwargs={
                        "experiment_case_reranking": experiment_case_reranking,
                        "experiment_case_baseline": experiment_case_baseline,
                        "experiment_baseline_similarity": similarity,
                    }
                )


def run_ablation_impressions_re_ranking_experiments(
    dask_interface: DaskInterface,
    ablation_re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    """
    Public method that instructs dask to run in dask workers the hyper-parameter tuning of the impressions discounting
    recommenders for the ablation study.

    Processes are always preferred than threads as the hyper-parameter tuning loop is probably not thread-safe.
    """
    for experiment_case_ablation_reranking in ablation_re_ranking_experiment_cases_interface.experiment_cases:
        for experiment_case_baseline in baseline_experiment_cases_interface.experiment_cases:
            experiment_can_be_tested = (
                experiment_case_ablation_reranking.benchmark == experiment_case_baseline.benchmark
                and experiment_case_ablation_reranking.hyper_parameter_tuning_parameters == experiment_case_baseline.hyper_parameter_tuning_parameters
            )

            if not experiment_can_be_tested:
                continue

            re_ranking_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
                experiment_case_ablation_reranking.benchmark
            ]
            re_ranking_recommender = commons.MAPPER_ABLATION_AVAILABLE_RECOMMENDERS[
                experiment_case_ablation_reranking.recommender
            ]

            baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
                experiment_case_baseline.recommender
            ]
            baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
                experiment_case_baseline.hyper_parameter_tuning_parameters
            ]

            similarities: Sequence[Optional[commons.T_SIMILARITY_TYPE]] = [None]
            if baseline_recommender.recommender in [
                recommenders.ItemKNNCFRecommender, recommenders.UserKNNCFRecommender
            ]:
                similarities = baseline_hyper_parameters.knn_similarity_types

            try_folded_recommenders = [False]
            if issubclass(baseline_recommender.recommender, BaseMatrixFactorizationRecommender):
                try_folded_recommenders = [True, False]

            for similarity in similarities:
                for try_folded_recommender in try_folded_recommenders:
                    dask_interface.submit_job(
                        job_key=(
                            f"_run_ablation_impressions_re_ranking_hyper_parameter_tuning"
                            f"|{experiment_case_ablation_reranking.benchmark.value}"
                            f"|{re_ranking_recommender.recommender.RECOMMENDER_NAME}"
                            f"|{baseline_recommender.recommender.RECOMMENDER_NAME}"
                            f"|{similarity}"
                            f"|{try_folded_recommender}"
                            f"|{uuid.uuid4()}"
                        ),
                        job_priority=(
                            re_ranking_benchmark.priority
                            * re_ranking_recommender.priority
                            * baseline_recommender.priority
                            * (100
                                if try_folded_recommender
                                else 10
                               )
                        ),
                        job_info={
                            "recommender": re_ranking_recommender.recommender.RECOMMENDER_NAME,
                            "baseline": baseline_recommender.recommender.RECOMMENDER_NAME,
                            "similarity": similarity,
                            "benchmark": re_ranking_benchmark.benchmark.value,
                            "try_folded_recommender": try_folded_recommender,
                        },
                        method=_run_ablation_impressions_re_ranking_hyper_parameter_tuning,
                        method_kwargs={
                            "experiment_case_ablation_reranking": experiment_case_ablation_reranking,
                            "experiment_case_baseline": experiment_case_baseline,
                            "experiment_baseline_similarity": similarity,
                            "try_folded_recommender": try_folded_recommender,
                        }
                    )


def run_signal_analysis_ablation_impressions_re_ranking_experiments(
    dask_interface: DaskInterface,
    ablation_re_ranking_experiment_cases_interface: commons.ExperimentCasesInterface,
    baseline_experiment_cases_interface: commons.ExperimentCasesInterface,
) -> None:
    """
    Public method that instructs dask to run in dask workers the hyper-parameter tuning of the impressions discounting
    recommenders for the ablation study while manually specifying the signals within impressions.

    Processes are always preferred than threads as the hyper-parameter tuning loop is probably not thread-safe.
    """
    for experiment_case_ablation_reranking in ablation_re_ranking_experiment_cases_interface.experiment_cases:
        for experiment_case_baseline in baseline_experiment_cases_interface.experiment_cases:
            experiment_can_be_tested = (
                experiment_case_ablation_reranking.benchmark == experiment_case_baseline.benchmark
                and experiment_case_ablation_reranking.hyper_parameter_tuning_parameters == experiment_case_baseline.hyper_parameter_tuning_parameters
            )

            if not experiment_can_be_tested:
                continue

            re_ranking_benchmark = commons.MAPPER_AVAILABLE_BENCHMARKS[
                experiment_case_ablation_reranking.benchmark
            ]
            re_ranking_recommender = commons.MAPPER_ABLATION_AVAILABLE_RECOMMENDERS[
                experiment_case_ablation_reranking.recommender
            ]

            baseline_recommender = commons.MAPPER_AVAILABLE_RECOMMENDERS[
                experiment_case_baseline.recommender
            ]
            baseline_hyper_parameters = commons.MAPPER_AVAILABLE_HYPER_PARAMETER_TUNING_PARAMETERS[
                experiment_case_baseline.hyper_parameter_tuning_parameters
            ]

            similarities: Sequence[Optional[commons.T_SIMILARITY_TYPE]] = [None]
            if baseline_recommender.recommender in [
                recommenders.ItemKNNCFRecommender, recommenders.UserKNNCFRecommender
            ]:
                similarities = baseline_hyper_parameters.knn_similarity_types

            try_folded_recommenders = [False]
            if issubclass(baseline_recommender.recommender, BaseMatrixFactorizationRecommender):
                try_folded_recommenders = [True, False]

            for similarity in similarities:
                for try_folded_recommender in try_folded_recommenders:
                    for signal_analysis_type in commons.SignalAnalysisType:
                        dask_interface.submit_job(
                            job_key=(
                                f"_run_ablation_impressions_re_ranking_hyper_parameter_tuning"
                                f"|{experiment_case_ablation_reranking.benchmark.value}"
                                f"|{re_ranking_recommender.recommender.RECOMMENDER_NAME}"
                                f"|{baseline_recommender.recommender.RECOMMENDER_NAME}"
                                f"|{similarity}"
                                f"|{try_folded_recommender}"
                                f"|{uuid.uuid4()}"
                            ),
                            job_priority=(
                                re_ranking_benchmark.priority
                                * re_ranking_recommender.priority
                                * baseline_recommender.priority
                                * (100
                                    if try_folded_recommender
                                    else 10
                                   )
                            ),
                            job_info={
                                "recommender": re_ranking_recommender.recommender.RECOMMENDER_NAME,
                                "baseline": baseline_recommender.recommender.RECOMMENDER_NAME,
                                "similarity": similarity,
                                "benchmark": re_ranking_benchmark.benchmark.value,
                                "try_folded_recommender": try_folded_recommender,
                            },
                            method=_run_signal_analysis_ablation_impressions_re_ranking_hyper_parameter_tuning,
                            method_kwargs={
                                "experiment_case_ablation_reranking": experiment_case_ablation_reranking,
                                "experiment_case_baseline": experiment_case_baseline,
                                "experiment_baseline_similarity": similarity,
                                "try_folded_recommender": try_folded_recommender,
                                "signal_analysis_type": signal_analysis_type,
                            }
                        )
