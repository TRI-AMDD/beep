import pandas as pd

from beep import PROTOCOL_PARAMETERS_DIR
from beep.features import helper_functions

from beep.features.featurizer import BEEPPerCycleFeaturizer



class HPPCResistanceVoltagePerCycle(BEEPPerCycleFeaturizer):
    DEFAULT_HYPERPARAMETERS = {
        "test_time_filter_sec": 1000000,
        "cycle_index_filter": 6,
        "soc_window": 8,
        "parameters_path": PROTOCOL_PARAMETERS_DIR
    }

    def validate(self):
        val, msg = helper_functions.check_diagnostic_validation(self.datapath)
        if val:
            conditions = []
            conditions.append(
                any(
                    [
                        "hppc" in x
                        for x in
                        self.datapath.diagnostic_summary.cycle_type.unique()
                    ]
                )
            )
            if all(conditions):
                return True, None
            else:
                return False, "HPPC conditions not met for this cycler run"
        else:
            return val, msg

    def create_features(self):
        # Filter out low cycle numbers at the end of the test, corresponding to the "final" diagnostic
        self.datapath.diagnostic_data = self.datapath.diagnostic_data[
            ~((self.datapath.diagnostic_data.test_time > self.hyperparameters[
                'test_time_filter_sec']) &
              (self.datapath.diagnostic_data.cycle_index < self.hyperparameters[
                  'cycle_index_filter']))
        ]
        self.datapath.diagnostic_data = self.datapath.diagnostic_data.groupby(
            ["cycle_index", "step_index", "step_index_counter"]
        ).filter(lambda x: ~x["test_time"].isnull().all())

        # Only hppc_resistance_features are able to be calculated without error.
        # Xiao Cui should be pulled in to understand the issue with the others features.

        # diffusion features
        #         diffusion_features = featurizer_helpers.get_diffusion_cycle_features(
        #             self.datapath,
        #         )

        # hppc resistance features
        hppc_resistance_features = helper_functions.get_hppc_resistance_cycle_features(
            self.datapath,
        )

        # the variance of ocv features
        #         hppc_ocv_features = featurizer_helpers.get_hppc_ocv_cycle_features(
        #             self.datapath,
        #         )

        # the v_diff features
        #         v_diff = featurizer_helpers.get_v_diff_cycle_features(
        #             self.datapath,
        #             self.hyperparameters["soc_window"],
        #             self.hyperparameters["parameters_path"]
        #         )

        # merge everything together as a final result dataframe
        self.features = pd.concat(
            [hppc_resistance_features,
             # hppc_ocv_features,
             # v_diff, #diffusion_features
             ], axis=1)
