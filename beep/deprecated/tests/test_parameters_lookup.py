import unittest





class TestBEEPDatapath(unittest.TestCase):
    """
    Tests common to all datapaths.
    """
    # based on RCRT.test_determine_structuring_parameters
    def test_determine_structuring_parameters(self):
        (v_range, resolution, nominal_capacity, full_fast_charge, diagnostic_available) = \
            self.datapath_diag_normal.determine_structuring_parameters()
        diagnostic_available_test = {
            "parameter_set": "Tesla21700",
            "cycle_type": ["reset", "hppc", "rpt_0.2C", "rpt_1C", "rpt_2C"],
            "length": 5,
            "diagnostic_starts_at": [
                1, 36, 141, 246, 351, 456, 561, 666, 771, 876, 981, 1086,
                1191,
                1296, 1401, 1506, 1611, 1716, 1821, 1926, 2031, 2136, 2241,
                2346,
                2451, 2556, 2661, 2766, 2871, 2976, 3081, 3186, 3291, 3396,
                3501,
                3606, 3628
            ]
        }

        self.assertEqual(v_range, [2.5, 4.2])
        self.assertEqual(resolution, 1000)
        self.assertEqual(nominal_capacity, 4.84)
        self.assertEqual(full_fast_charge, 0.8)
        self.assertEqual(diagnostic_available, diagnostic_available_test)
        (
            v_range,
            resolution,
            nominal_capacity,
            full_fast_charge,
            diagnostic_available,
        ) = self.datapath_diag_misplaced.determine_structuring_parameters()
        diagnostic_available_test = {
            "parameter_set": "Tesla21700",
            "cycle_type": ["reset", "hppc", "rpt_0.2C", "rpt_1C", "rpt_2C"],
            "length": 5,
            "diagnostic_starts_at": [1, 36, 141, 220, 255]
        }
        self.assertEqual(v_range, [2.5, 4.2])
        self.assertEqual(resolution, 1000)
        self.assertEqual(nominal_capacity, 4.84)
        self.assertEqual(full_fast_charge, 0.8)
        self.assertEqual(diagnostic_available, diagnostic_available_test)