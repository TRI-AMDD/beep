# Copyright 2019 Toyota Research Institute. All rights reserved.
import json
import os
import unittest
import numpy as np
from beep import TEST_FILE_DIR
from sklearn.decomposition import PCA
from beep.principal_components import PrincipalComponents, pivot_data

class PrincipalComponentsTest(unittest.TestCase):
    def setUp(self):
        self.processed_run_path = os.path.join(TEST_FILE_DIR, "2017-06-30_2C-10per_6C_CH10_structure.json")
        self.cycles_to_pca = np.linspace(20, 120, 20, dtype=int)
        self.cycles_to_test = np.linspace(121, 131, 6, dtype=int)
        json_obj = {
            "file_list": [self.processed_run_path],
            "run_list": [1]
        }
        json_string = json.dumps(json_obj)
        self.pc = PrincipalComponents.from_interpolated_data(json_string, cycles_to_pca=self.cycles_to_pca)

    def test_pivot_data(self):
        json_obj = {
            "file_list": [self.processed_run_path],
            "run_list": [1]
        }
        json_string = json.dumps(json_obj)
        df_to_pca = pivot_data(json_string, 'discharge_capacity', 'voltage', self.cycles_to_pca)
        self.assertEqual(df_to_pca.shape, (len(self.cycles_to_pca), 1000))

    def test_fit(self):
        self.assertIsInstance(self.pc.pca, PCA)
        self.assertEqual(self.pc.min_components, 4)

    def test_get_pca_embeddings(self):
        json_obj = {
            "file_list": [self.processed_run_path],
            "run_list": [1]
        }
        json_string = json.dumps(json_obj)
        df_to_pca = pivot_data(json_string, 'discharge_capacity', 'voltage', self.cycles_to_test)
        pca_embeddings = self.pc.get_pca_embeddings(df_to_pca)
        self.assertEqual(pca_embeddings.shape, (len(self.cycles_to_test), self.pc.n_components))


    def test_get_pca_reconstruction(self):
        """
        Method to inverse transform PCA embeddings to reconstruct data
        """
        json_obj = {
            "file_list": [self.processed_run_path],
            "run_list": [1]
        }
        json_string = json.dumps(json_obj)
        df_to_pca = pivot_data(json_string, 'discharge_capacity', 'voltage', self.cycles_to_test)
        pca_embeddings = self.pc.get_pca_embeddings(df_to_pca)
        pca_reconstruction = self.pc.get_pca_reconstruction(pca_embeddings)
        self.assertEqual(pca_reconstruction.shape, (len(self.cycles_to_test), 1000))

    def test_get_reconstruction_errors(self):
        json_obj = {
            "file_list": [self.processed_run_path],
            "run_list": [1]
        }
        json_string = json.dumps(json_obj)
        df_to_pca = pivot_data(json_string, 'discharge_capacity', 'voltage', self.cycles_to_test)
        reconstruction_errors, outliers = self.pc.get_reconstruction_error_outliers(df_to_pca, threshold=1.5)
        self.assertAlmostEqual(reconstruction_errors[0], 0.002553278, places=8)
        self.assertTrue(outliers[0])

if __name__ == "__main__":
    unittest.main()
