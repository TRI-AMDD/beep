import os

from beep import PROTOCOL_PARAMETERS_DIR

def determine_structuring_parameters(
        self,
        v_range=None,
        resolution=1000,
        nominal_capacity=1.1,
        full_fast_charge=0.8,
        parameters_path=PROTOCOL_PARAMETERS_DIR,
    ):
        """
        Method for determining what values to use to convert raw run into processed run.


        Args:
            v_range ([float, float]): voltage range for interpolation
            resolution (int): resolution for interpolation
            nominal_capacity (float): nominal capacity for summary stats
            full_fast_charge (float): full fast charge for summary stats
            parameters_path (str): path to parameters files

        Returns:
            v_range ([float, float]): voltage range for interpolation
            resolution (int): resolution for interpolation
            nominal_capacity (float): nominal capacity for summary stats
            full_fast_charge (float): full fast charge for summary stats

        """
        if not parameters_path or not os.path.exists(parameters_path):
            raise FileNotFoundError(f"Parameters path {parameters_path} does not exist!")

        run_parameter, all_parameters = parameters_lookup.get_protocol_parameters(
            self.paths["raw"], parameters_path
        )

        # Logic for interpolation variables and diagnostic cycles
        diagnostic_available = False
        if run_parameter is not None:
            if {"capacity_nominal"}.issubset(run_parameter.columns.tolist()):
                nominal_capacity = run_parameter["capacity_nominal"].iloc[0]
            if {"discharge_cutoff_voltage", "charge_cutoff_voltage"}.issubset(
                    run_parameter.columns):
                v_range = [
                    all_parameters["discharge_cutoff_voltage"].min(),
                    all_parameters["charge_cutoff_voltage"].max(),
                ]
            if {"diagnostic_type", "diagnostic_start_cycle", "diagnostic_interval"}.issubset(run_parameter.columns):
                if run_parameter["diagnostic_type"].iloc[0] == "HPPC+RPT":
                    hppc_rpt = ["reset", "hppc", "rpt_0.2C", "rpt_1C", "rpt_2C"]
                    hppc_rpt_len = 5
                    initial_diagnostic_at = [1, 1 + run_parameter["diagnostic_start_cycle"].iloc[0] + 1 * hppc_rpt_len]
                    # Calculate the number of steps present for each cycle in the diagnostic as the pattern for
                    # the diagnostic. If this pattern of steps shows up at the end of the file, that indicates
                    # the presence of a final diagnostic
                    diag_0_pattern = [len(self.raw_data[self.raw_data.cycle_index == x].step_index.unique())
                                      for x in range(initial_diagnostic_at[0], initial_diagnostic_at[0] + hppc_rpt_len)]
                    diag_1_pattern = [len(self.raw_data[self.raw_data.cycle_index == x].step_index.unique())
                                      for x in range(initial_diagnostic_at[1], initial_diagnostic_at[1] + hppc_rpt_len)]
                    # Find the steps present in the reset cycles for the first and second diagnostic
                    diag_0_steps = set(self.raw_data[self.raw_data.cycle_index ==
                                       initial_diagnostic_at[0]].step_index.unique())
                    diag_1_steps = set(self.raw_data[self.raw_data.cycle_index ==
                                       initial_diagnostic_at[1]].step_index.unique())
                    diagnostic_starts_at = []
                    for cycle in self.raw_data.cycle_index.unique():
                        steps_present = set(self.raw_data[
                                                self.raw_data.cycle_index == cycle].step_index.unique())
                        cycle_pattern = [len(self.raw_data[
                                                 self.raw_data.cycle_index == x].step_index.unique())
                                         for x in
                                         range(cycle, cycle + hppc_rpt_len)]
                        if steps_present == diag_0_steps or steps_present == diag_1_steps:
                            diagnostic_starts_at.append(cycle)
                        # Detect final diagnostic if present in the data
                        elif cycle >= (
                                self.raw_data.cycle_index.max() - hppc_rpt_len - 1) and \
                                (cycle_pattern == diag_0_pattern or cycle_pattern == diag_1_pattern):
                            diagnostic_starts_at.append(cycle)

                    diagnostic_available = {
                        "parameter_set":
                            run_parameter["diagnostic_parameter_set"].iloc[0],
                        "cycle_type": hppc_rpt,
                        "length": hppc_rpt_len,
                        "diagnostic_starts_at": diagnostic_starts_at,
                    }

        return (
            v_range,
            resolution,
            nominal_capacity,
            full_fast_charge,
            diagnostic_available,
        )