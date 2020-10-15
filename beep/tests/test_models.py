# Copyright [2020] [Toyota Research Institute]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests related to feature generation"""

import unittest
import os
import json
import numpy as np
from glob import glob
from beep import MODEL_DIR, ENVIRONMENT
from beep.run_model import (
    DegradationModel,
    process_file_list_from_json,
    get_project_name_from_list,
)
from monty.serialization import loadfn
from monty.tempfile import ScratchDir


TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")
SINGLE_TASK_FEATURES_PATH = os.path.join(
    TEST_FILE_DIR, "feature_jsons_for_training_model", "single_task"
)
MULTI_TASK_FEATURES_PATH = os.path.join(
    TEST_FILE_DIR, "feature_jsons_for_training_model", "multi_task"
)


class TestRunModel(unittest.TestCase):
    def setUp(self):
        pass

    def test_model_training_and_serialization(self):
        # tests for model training and serialization
        featurized_jsons = glob(
            os.path.join(SINGLE_TASK_FEATURES_PATH, "*features.json")
        )
        with ScratchDir(".") as scratch_dir:
            json_obj = {
                "file_list": featurized_jsons,
                "run_list": list(range(len(featurized_jsons))),
            }
            json_string = json.dumps(json_obj)
            model = DegradationModel.train(
                json_string,
                dataset_id=2,
                model_type="linear",
                regularization_type="elasticnet",
                model_name="test_model",
                hyperparameters={},
            )
            model.serialize(processed_dir=scratch_dir)
            serialized_model_reloaded = DegradationModel.from_serialized_model(
                model_dir=scratch_dir, serialized_model=model.name + ".model"
            )
            self.assertIsInstance(serialized_model_reloaded, DegradationModel)

    def test_multi_task_model_training(self):
        featurized_jsons = glob(
            os.path.join(MULTI_TASK_FEATURES_PATH, "*features.json")
        )
        with ScratchDir(".") as scratch_dir:
            json_obj = {
                "file_list": featurized_jsons,
                "run_list": list(range(len(featurized_jsons))),
            }
            json_string = json.dumps(json_obj)

            model = DegradationModel.train(
                json_string,
                dataset_id=2,
                model_type="linear",
                regularization_type="elasticnet",
                model_name="test_model",
                hyperparameters={},
            )
            model.serialize(processed_dir=scratch_dir)
            serialized_model_reloaded = DegradationModel.from_serialized_model(
                model_dir=scratch_dir, serialized_model=model.name + ".model"
            )
            self.assertGreater(
                len(serialized_model_reloaded.model["confidence_bounds"]), 1
            )
            self.assertIsInstance(serialized_model_reloaded, DegradationModel)

    def test_multi_task_prediction_list_to_json(self):
        featurized_jsons = glob(
            os.path.join(MULTI_TASK_FEATURES_PATH, "*features.json")
        )
        json_obj = {
            "file_list": featurized_jsons,
            "run_list": list(range(len(featurized_jsons))),
        }
        json_string = json.dumps(json_obj)
        newjsonpaths = process_file_list_from_json(
            json_string,
            model_dir=MODEL_DIR,
            processed_dir=MULTI_TASK_FEATURES_PATH,
            predict_only=True,
        )
        reloaded = json.loads(newjsonpaths)

        prediction_reloaded = loadfn(reloaded["file_list"][0])
        self.assertIsInstance(prediction_reloaded["cycle_number"], np.ndarray)
        self.assertGreater(len(prediction_reloaded["cycle_number"]), 1)

        # Testing error output
        self.assertIsInstance(prediction_reloaded["fractional_error"], np.ndarray)
        self.assertEqual(
            len(prediction_reloaded["fractional_error"]), 1
        )  # for now just a single fractional error
        predictions = glob(os.path.join(MULTI_TASK_FEATURES_PATH, "*predictions.json"))
        for file in predictions:
            os.remove(file)

    def test_serialized_prediction(self):
        # Testing nominal capacity calculation
        feature_json_path = os.path.join(
            TEST_FILE_DIR, "2017-06-30_2C-10per_6C_CH10_full_model_multi_features.json"
        )
        json_obj = {
            "file_list": [feature_json_path, feature_json_path],
            "run_list": [0, 1],
        }
        json_string = json.dumps(json_obj)
        newjsonpaths = process_file_list_from_json(
            json_string, model_dir=MODEL_DIR, processed_dir=TEST_FILE_DIR
        )
        reloaded = json.loads(newjsonpaths)
        prediction_reloaded = loadfn(reloaded["file_list"][0])

        features = loadfn(feature_json_path)
        self.assertEqual(features.nominal_capacity, 1.0628421000000001)
        self.assertFalse(
            (
                prediction_reloaded["discharge_capacity"]
                - np.around(np.arange(0.98, 0.78, -0.03), 2) * features.nominal_capacity
            ).any()
        )
        os.remove(
            os.path.join(
                TEST_FILE_DIR,
                "2017-06-30_2C-10per_6C_CH10_full_model_multi_predictions.json",
            )
        )

    def test_single_task_prediction_list_to_json(self):
        featurized_jsons = glob(
            os.path.join(SINGLE_TASK_FEATURES_PATH, "*features.json")
        )
        json_obj = {
            "file_list": featurized_jsons,
            "run_list": list(range(len(featurized_jsons))),
        }
        json_string = json.dumps(json_obj)
        newjsonpaths = process_file_list_from_json(
            json_string,
            model_dir=MODEL_DIR,
            processed_dir=SINGLE_TASK_FEATURES_PATH,
            predict_only=True,
        )
        reloaded = json.loads(newjsonpaths)

        # Ensure first is correct
        prediction_reloaded = loadfn(reloaded["file_list"][0])
        self.assertIsInstance(prediction_reloaded["cycle_number"], np.ndarray)
        # Testing error output
        self.assertIsInstance(prediction_reloaded["fractional_error"], np.ndarray)

        predictions = glob(os.path.join(SINGLE_TASK_FEATURES_PATH, "*predictions.json"))
        for file in predictions:
            os.remove(file)


class TestHelperFunctions(unittest.TestCase):
    def test_get_project_name_from_list(self):
        file_list = [
            "data-share/predictions/PredictionDiagnostics_000100_003022_predictions.json",
            "data-share/predictions/PredictionDiagnostics_000101_003022_predictions.json",
            "data-share/predictions/PredictionDiagnostics_000102_003022_predictions.json",
        ]
        project_name = get_project_name_from_list(file_list)
        self.assertEqual(project_name, "PredictionDiagnostics")
