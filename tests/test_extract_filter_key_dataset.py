"""Materialize hdf5 filter keys as standalone datasets."""
from scripts.utils.extract_filter_key_dataset import output_label


def test_output_label_drops_percent_token() -> None:
    assert output_label(filter_key="50_percent_train", num_demos=90) == "train_90"
    assert output_label(filter_key="20_percent_valid", num_demos=4) == "valid_4"


def test_output_label_keeps_count_when_only_percent() -> None:
    assert output_label(filter_key="50_percent", num_demos=100) == "100"


def test_output_label_strips_quality_prefix() -> None:
    assert output_label(filter_key="better_train", num_demos=90, strip_prefix="better_") == "train_90"
    assert (
        output_label(
            filter_key="better_operator_1_train", num_demos=45, strip_prefix="better_"
        )
        == "operator_1_train_45"
    )


def test_output_label_leaves_non_matching_prefix_alone() -> None:
    assert output_label(filter_key="better_train", num_demos=90, strip_prefix="okay_") == "better_train_90"
