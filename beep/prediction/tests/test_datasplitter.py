import os
import unittest
from beep.tests.constants import TEST_FILE_DIR
from beep.features.base import BEEPFeatureMatrix
from beep.prediction.beep_data_splitter import BEEPDataSplitter


class TestBEEPDataSplitter(unittest.TestCase):

    def setUp(self):
        bfm_path = os.path.join(TEST_FILE_DIR, "feature_matrix.json")
        # bfm_dict = json.load(open(bfm_path,'r'))
        self.sample_feature_matrix = BEEPFeatureMatrix.from_json_file(bfm_path)
        self.features = ["::".join(c.split("::")[:2]) for c in self.sample_feature_matrix.matrix.columns if not any(
            [feat in c for feat in ["DiagnosticProperties", "ChargingProtocol", "ExclusionCriteria"]])]

    def test_datasplitter(self):

        datasplitter = BEEPDataSplitter(
            feature_matrix=self.sample_feature_matrix,
            features=self.features,
            targets=["rpt_0.2Cdischarge_capacity0.8_real_regular_throughput::DiagnosticProperties"],
            n_splits=5,
            )

        datasplitter.split()

        self.assertEqual(len(datasplitter.datasets), 5)
        self.assertEqual(len(datasplitter.datasets[0].train_X.columns), 54)

        self.assertAlmostEqual(
            datasplitter.datasets[0].train_X["abs_charging_capacity::DiagnosticSummaryStats"].iloc[0], 1.2092409629538559, places=6)

        self.assertTrue(all([d in self.features for d in datasplitter.datasets[0].train_X.columns]))

        n_cleaned_samples = 6
        for dataset in datasplitter.datasets:
            self.assertEqual(len(dataset.train_X.index) + len(dataset.test_X.index), n_cleaned_samples)
            self.assertEqual(len(dataset.train_y.index) + len(dataset.test_y.index), n_cleaned_samples)

    def test_split_on_charging_protocol(self):

        split_columns = ["charge_constant_current_1", "charge_constant_current_2"]
        split_columns = [f"{c}::ChargingProtocol" for c in split_columns]

        datasplitter = BEEPDataSplitter(
            feature_matrix=self.sample_feature_matrix,
            features=self.features,
            targets=["rpt_0.2Cdischarge_capacity0.8_real_regular_throughput::DiagnosticProperties"],
            n_splits=2,
            split_columns=split_columns
            )

        datasplitter.split()

        self.assertEqual(len(datasplitter.datasets), 2)
        self.assertEqual(len(datasplitter.datasets[0].train_X.columns), 54)

        self.assertAlmostEqual(
            datasplitter.datasets[0].train_X["abs_charging_capacity::DiagnosticSummaryStats"].iloc[0], 0.955712306816296, places=6)

        n_cleaned_samples = 6
        for dataset in datasplitter.datasets:
            self.assertEqual(len(dataset.train_X.index) + len(dataset.test_X.index), n_cleaned_samples)
            self.assertEqual(len(dataset.train_y.index) + len(dataset.test_y.index), n_cleaned_samples)

            # Test that unique values of split_columns are not split between train and test sets

            # Find the unique combinations of split combination values in train and test sets
            split_column_values_train = self.sample_feature_matrix.matrix.loc[dataset.train_X.index][split_columns]
            unique_split_column_values_train = split_column_values_train.apply(
                lambda x: "::".join([str(x[s]) for s in split_columns]), axis=1).unique()
            unique_split_column_values_train = set(unique_split_column_values_train)

            split_column_values_test = self.sample_feature_matrix.matrix.loc[dataset.test_X.index][split_columns]
            unique_split_column_values_test = split_column_values_test.apply(
                lambda x: "::".join([str(x[s]) for s in split_columns]), axis=1).unique()
            unique_split_column_values_test = set(unique_split_column_values_test)

            self.assertTrue(unique_split_column_values_train.isdisjoint(unique_split_column_values_test))

    def test_exclude_cells(self):

        exclusion_columns = ["is_above_first_n_cycles_throughput",
                             "is_below_fractional_capacity_at_EOT",
                             "is_above_equivalent_full_cycles_at_EOL",
                             "is_not_early_CV",
                             ]

        exclusion_columns = [f"{c}::ExclusionCriteria" for c in exclusion_columns]

        datasplitter = BEEPDataSplitter(
            feature_matrix=self.sample_feature_matrix,
            features=self.features,
            targets=["rpt_0.2Cdischarge_capacity0.8_real_regular_throughput::DiagnosticProperties"],
            n_splits=3,
            exclusion_columns=exclusion_columns,
            )

        # 6 samples go in, 3 should pass exclusion check
        n_cleaned_samples = 3

        self.assertEqual(len(datasplitter.X), n_cleaned_samples)
        datasplitter.split()

        self.assertEqual(len(datasplitter.datasets), 3)

        self.assertEqual(len(datasplitter.datasets[0].train_X.columns), 54)

        self.assertAlmostEqual(
            datasplitter.datasets[0].train_X["abs_charging_capacity::DiagnosticSummaryStats"].iloc[0], 0.955712306816296, places=6)

        for dataset in datasplitter.datasets:
            self.assertEqual(len(dataset.train_X.index) + len(dataset.test_X.index), n_cleaned_samples)
            self.assertEqual(len(dataset.train_y.index) + len(dataset.test_y.index), n_cleaned_samples)

    def test_datasplitter_serialization(self):
        datasplitter = BEEPDataSplitter(
            feature_matrix=self.sample_feature_matrix,
            features=self.features,
            targets=["rpt_0.2Cdischarge_capacity0.8_real_regular_throughput::DiagnosticProperties"],
            n_splits=5,
            )

        featurized_datasplitter_path = os.path.join(TEST_FILE_DIR, "test_datasplitter.json")
        datasplitter.to_json_file(featurized_datasplitter_path)

        reloaded_datasplitter = BEEPDataSplitter.from_json_file(featurized_datasplitter_path)

        self.assertEqual(len(reloaded_datasplitter.datasets), 5)
        self.assertEqual(len(reloaded_datasplitter.datasets[0].train_X.columns), 54)

        self.assertAlmostEqual(
            datasplitter.datasets[0].train_X["abs_charging_capacity::DiagnosticSummaryStats"].iloc[0], 1.2092409629538559, places=6)

        n_cleaned_samples = 6
        for dataset in reloaded_datasplitter.datasets:
            self.assertEqual(len(dataset.train_X.index) + len(dataset.test_X.index), n_cleaned_samples)
            self.assertEqual(len(dataset.train_y.index) + len(dataset.test_y.index), n_cleaned_samples)
