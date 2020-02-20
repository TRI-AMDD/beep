# Copyright 2019 Toyota Research Institute. All rights reserved.
"""Unit tests related to feature generation"""

import unittest
import os
import json
import boto3

import numpy as np
from botocore.exceptions import NoRegionError, NoCredentialsError

from beep_ep import TEST_FILE_DIR
from beep_ep.structure import RawCyclerRun, ProcessedCyclerRun
from beep_ep.featurize import DegradationPredictor, process_file_list_from_json
from monty.serialization import dumpfn, loadfn

processed_cycler_file = "2017-06-30_2C-10per_6C_CH10_structure.json"
processed_cycler_file_insuf = "structure_insufficient.json"
maccor_file_w_diagnostics = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000020_CH71.071")

BIG_FILE_TESTS = os.environ.get("BEEP_BIG_TESTS", False)
SKIP_MSG = "Tests requiring large files with diagnostic cycles are disabled, set BIG_FILE_TESTS to run full tests"


class TestFeaturizer(unittest.TestCase):
    def setUp(self):
        # Setup events for testing
        try:
            kinesis = boto3.client('kinesis')
            response = kinesis.list_streams()
            self.events_mode = "test"
        except NoRegionError or NoCredentialsError as e:
            self.events_mode = "events_off"

    def test_feature_generation_full_model(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, processed_cycler_file)
        predictor = DegradationPredictor.from_processed_cycler_run_file(processed_cycler_run_path,
                                                                        features_label='full_model')
        self.assertEqual(len(predictor.X), 1)  # just test if works for now
        # Ensure no NaN values
        self.assertFalse(np.any(predictor.X.isnull()))

    def test_feature_label_full_model(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, processed_cycler_file)
        predictor = DegradationPredictor.from_processed_cycler_run_file(processed_cycler_run_path,
                                                                        features_label='full_model')
        self.assertEqual(predictor.feature_labels[4], "charge_time_cycles_1:5")  

    def test_feature_serialization(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, processed_cycler_file)
        predictor = DegradationPredictor.from_processed_cycler_run_file(processed_cycler_run_path,
                                                                        prediction_type = 'multi',
                                                                        features_label='full_model')
        #    import nose
    #    nose.tools.set_trace()
        dumpfn(predictor, os.path.join(TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_CH29_features_predict_only.json"))
        predictor_reloaded = loadfn(os.path.join(TEST_FILE_DIR,
                                                 "2017-12-04_4_65C-69per_6C_CH29_features_predict_only.json"))
        self.assertIsInstance(predictor_reloaded, DegradationPredictor)
        # test nominal capacity is being generated
        self.assertEqual(predictor_reloaded.nominal_capacity, 1.0628421000000001)

    def test_feature_serialization_for_training(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, processed_cycler_file)
        predictor = DegradationPredictor.from_processed_cycler_run_file(processed_cycler_run_path,
                                                                        features_label='full_model', predict_only=False)
        dumpfn(predictor, os.path.join(TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_CH29_features.json"))
        predictor_reloaded = loadfn(os.path.join(TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_CH29_features.json"))
        self.assertIsInstance(predictor_reloaded, DegradationPredictor)

    @unittest.skipUnless(BIG_FILE_TESTS, SKIP_MSG)
    def test_diagnostic_feature_generation(self):
        os.environ['BEEP_EP_ROOT'] = TEST_FILE_DIR
        maccor_file_w_parameters = os.path.join(TEST_FILE_DIR, "PredictionDiagnostics_000136_00002D.037")
        raw_run = RawCyclerRun.from_file(maccor_file_w_parameters)
        v_range, resolution, nominal_capacity, full_fast_charge, diagnostic_available = \
            raw_run.determine_structuring_parameters()
        pcycler_run = ProcessedCyclerRun.from_raw_cycler_run(raw_run,
                                                             diagnostic_available=diagnostic_available)
        predictor = DegradationPredictor.init_full_model(pcycler_run,
                                                         predict_only=False,
                                                         mid_pred_cycle=11,
                                                         final_pred_cycle=12,
                                                         diagnostic_features=True)
        diagnostic_feature_label = predictor.feature_labels[-1]
        self.assertEqual(diagnostic_feature_label, "median_diagnostic_cycles_discharge_capacity")
        np.testing.assert_almost_equal(predictor.X[diagnostic_feature_label][0], 4.2327690032,  decimal=8)

    def test_feature_generation_list_to_json(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, processed_cycler_file)
        # Create dummy json obj
        json_obj = {
                    "mode": self.events_mode,
                    "file_list": [processed_cycler_run_path, processed_cycler_run_path],
                    'run_list': [0, 1]
                    }
        json_string = json.dumps(json_obj)

        newjsonpaths = process_file_list_from_json(json_string, processed_dir=TEST_FILE_DIR)
        reloaded = json.loads(newjsonpaths)

        # Check that at least strings are output
        self.assertIsInstance(reloaded['file_list'][-1], str)

        # Ensure first is correct
        predictor_reloaded = loadfn(reloaded['file_list'][0])
        self.assertIsInstance(predictor_reloaded, DegradationPredictor)
        self.assertEqual(predictor_reloaded.nominal_capacity, 1.0628421000000001)

    def test_insufficient_data_file(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, processed_cycler_file_insuf)

        json_obj = {
                    "mode": self.events_mode,
                    "file_list": [processed_cycler_run_path],
                    'run_list': [1]
                    }
        json_string = json.dumps(json_obj)

        json_path = process_file_list_from_json(json_string, processed_dir=TEST_FILE_DIR)
        output_obj = json.loads(json_path)
        self.assertEqual(output_obj['result_list'][0],'incomplete')
        self.assertEqual(output_obj['message_list'][0]['comment'],'Insufficient data for featurization')
