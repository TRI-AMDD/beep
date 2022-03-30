import pandas as pd

from beep import PROTOCOL_PARAMETERS_DIR
from beep.features import featurizer_helpers
from beep.features.featurizer import BEEPEarlyCyclesFeaturizer


class HPPCResistanceVoltage(BEEPEarlyCyclesFeaturizer):
    DEFAULT_HYPERPARAMETERS = {
        "test_time_filter_sec": 1000000,
        "cycle_index_filter": 6,
        "diag_pos": 1,
        "soc_window": 8,
        "parameters_path": PROTOCOL_PARAMETERS_DIR
    }

    def validate(self):
        val, msg = featurizer_helpers.check_diagnostic_validation(self.datapath)
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

        # diffusion features
        diffusion_features = featurizer_helpers.get_diffusion_early_features(
            self.datapath,
        )

        hppc_r = pd.DataFrame()
        # the 9 by 6 dataframe
        df_dr = featurizer_helpers.get_dr_df(
            self.datapath, self.hyperparameters["diag_pos"]
        )
        # transform this dataframe to be 1 by 54
        columns = df_dr.columns
        for column in columns:
            for r in range(len(df_dr[column])):
                name = column + str(r)
                hppc_r[name] = [df_dr[column][r]]

        # the variance of ocv features
        hppc_ocv = featurizer_helpers.get_hppc_ocv(
            self.datapath,
            self.hyperparameters["diag_pos"],
            parameters_path=self.hyperparameters["parameters_path"]
        )

        # the v_diff features
        v_diff = featurizer_helpers.get_v_diff(
            self.datapath,
            self.hyperparameters["diag_pos"],
            self.hyperparameters["soc_window"],
            self.hyperparameters["parameters_path"]
        )

        # merge everything together as a final result dataframe
        self.features = pd.concat(
            [hppc_r, hppc_ocv, v_diff, diffusion_features], axis=1)