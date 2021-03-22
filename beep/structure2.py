













if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # test_arbin_path = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv"
    # test_maccor_path_w_diagnostics = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/xTESLADIAG_000020_CH71.071"
    #
    # from beep.structure import RawCyclerRun as rcrv1, \
    #     ProcessedCyclerRun as pcrv1
    #
    # # rcr = rcrv1.from_arbin_file(test_arbin_path)
    # # rcr.data.to_csv("BEEPDatapath_arbin_memloaded.csv")
    # # with open("tests/test_files/BEEPDatapath_arbin_metadata_memloaded.json", "w") as f:
    # #     json.dump(rcr.metadata, f)
    #
    #
    #
    #
    # # test_maccor_paused_path = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000151_paused.052"
    # # rcr = rcrv1.from_maccor_file(test_maccor_paused_path, include_eis=False)
    # # rcr.data.to_csv("BEEPDatapath_maccor_paused_memloaded.csv")
    # # with open("tests/test_files/BEEPDatapath_maccor_paused_metadata_memloaded.json", "w") as f:
    # #     json.dump(rcr.metadata, f)
    #
    #
    # test_maccor_paused_path = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000151_test.052"
    # rcr = rcrv1.from_maccor_file(test_maccor_paused_path, include_eis=False)
    # rcr.data.to_csv("tests/test_files/BEEPDatapath_maccor_timestamp_memloaded.csv")
    # with open("tests/test_files/BEEPDatapath_maccor_timestamp_metadata_memloaded.json", "w") as f:
    #     json.dump(rcr.metadata, f)
    #
    #
    # # rcr = rcrv1.from_maccor_file(filename=test_maccor_path_w_diagnostics, include_eis=False)
    # # rcr.data.to_csv("BEEPDatapath_maccor_w_diagnostic_memloaded.csv")
    # #
    # # with open("BEEPDatapath_maccor_with_diagnostic_metadata_memloaded.json", "w") as f:
    # #     json.dump(rcr.metadata, f)
    #
    #
    # print(rcr.metadata)
    #
    # print(rcr.data)



    # maccor = MaccorDatapath.from_file("/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000109_tztest.010")
    # maccor.raw_data.to_csv("tests/test_files/BEEPDatapath_maccor_parameterized.csv")
    # with open("tests/test_files/BEEPDatapath_maccor_parameterized_metadata_memloaded.json", "w") as f:
    #     json.dump(maccor.metadata.raw, f)
    # maccor.structure()
    # print(maccor.get_cycle_life())

    # self.maccor_file_diagnostic_normal = os.path.join(
    #     TEST_FILE_DIR, "PreDiag_000287_000128short.092"
    # )
    # self.maccor_file_diagnostic_misplaced = os.path.join(
    #     TEST_FILE_DIR, "PreDiag_000412_00008Fshort.022"

    for d, f in {"diagnostic_normal":"PreDiag_000287_000128short.092", "diagnostic_misplaced":"PreDiag_000412_00008Fshort.022"}.items():
        maccor = MaccorDatapath.from_file(os.path.join("/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/", f))
        maccor.raw_data.to_csv(f"tests/test_files/BEEPDatapath_maccor_{d}_memloaded.csv")
        with open(f"tests/test_files/BEEPDatapath_maccor_{d}_metadata_memloaded.json", "w") as fi:
            json.dump(maccor.metadata.raw, fi)






    raise ValueError
    #
    # pcr = pcrv1.from_raw_cycler_run(rcr)
    #
    # print(pcr.cycles_interpolated)
    #
    # ad = ArbinDatapath.from_file(test_arbin_path)
    #
    # print(ad.raw_data)
    #
    # df = ad.interpolate_cycles(v_range=None, resolution=1000,
    #                            diagnostic_available=False,
    #                            charge_axis="charge_capacity",
    #                            discharge_axis="discharge_capacity")
    #
    # print(df)

    # todo: only processed_cycler run MSONable is used

    from beep.validate import ValidatorBeep, SimpleValidator


    df1 = rcr.data
    df2 = pd.read_csv(test_arbin_path, index_col=0)

    # vb = ValidatorBeep()
    # print(vb.validate_arbin_dataframe(df1))
    # print(vb.validate_arbin_dataframe(df2))

    # print(vb.errors)

    sv = SimpleValidator()
