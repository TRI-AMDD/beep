# Inspect


BEEP inspect is a debugging and analysis command which can be used to examine any 
serialized beep object directly from the command line.

The objects that can be inspected are:

- Raw cycler files compatible with BEEP, which will be ingested and represented as a `BEEPDatapath`. [Example: Inpsect Raw Files](#inspect-raw-files)
- Structured cycler files serialized by BEEP to disk as json, represented as a `BEEPDatapath`. [Example: Inspect Structured Files](#inspect-structured-files)
- Feature matrices serialized to disk as json. [Example: Inspect Feature Matrices](#inspect-feature-matrices)
- Individual `BEEPFeaturizers` serialized to disk as json. [Example: Inspect Featurizers](#inspect)

## Inspect help dialog

```shell
$: beep inspect --help

Usage: beep inspect [OPTIONS] FILE

  View BEEP files for debugging and analysis.

Options:
  --help  Show this message and exit.

```





## Inspect Raw Files

Example:

```shell
S: beep inspect PreDiag_000287_000128.092

2021-09-22 16:01:33 DEBUG    Loaded potential raw file /Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000287_000128.092 as Datapath.
2021-09-22 16:01:34 INFO     Loaded /Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000287_000128.092 as type <class 'beep.structure.maccor.MaccorDatapath'>.


BEEP Datapath: /Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000287_000128.092



Semiunique id: 'barcode:000128-channel:92-protocol:PreDiag_000287.000-schema:/Users/ardunn/alex/tri/code/beep/beep/validation_schemas/schema-maccor-2170.yaml-structured:False-legacy:False-raw_path:/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000287_000128.092-structured_path:None'

File paths
{'metadata': '/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000287_000128.092',
 'raw': '/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000287_000128.092'}

File metadata:
{'_today_datetime': '12/17/2019',
 'barcode': '000128',
 'channel_id': 92,
 'filename': 'C:\\Users\\Maccor Tester User\\Documents\\Backup\\STANFORD '
             'LOANER #1\\STANFORD LOANER #1\\PreDiag_000287_000128.092',
 'protocol': 'PreDiag_000287.000',
 'start_datetime': '12/17/2019'}

Validation schema: /Users/ardunn/alex/tri/code/beep/beep/validation_schemas/schema-maccor-2170.yaml

Structuring parameters:
{}

Structured attributes:

structured_summary:
        No object.

structured_data:
        No object.

diagnostic_data:
        No object.

diagnostic_summary:
        No object.

raw_data:
        data_point  cycle_index  step_index   test_time     step_time  _capacity    _energy   current   voltage _state _ending_status            date_time  loop1  loop2  loop3  loop4  ac_impedence  internal_resistance  _wf_chg_cap  _wf_dis_cap  ...  _var2  _var3  _var4  _var5  _var6  _var7  _var8  _var9  _var10  _var11  _var12  _var13  _var14  _var15  charge_capacity  discharge_capacity  charge_energy  discharge_energy              date_time_iso  temperature
0                1            0           1        0.00      0.000000   0.000000   0.000000  0.000000  3.458076      R              0  12/17/2019 09:51:51      0      0      0      0           0.0                  0.0          NaN          NaN  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0         0.000000            0.000000       0.000000          0.000000  2019-12-17T17:51:51+00:00          NaN
1                2            0           1       30.00     30.000000   0.000000   0.000000  0.000000  3.457999      R              1  12/17/2019 09:52:20      0      0      0      0           0.0                  0.0          NaN          NaN  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0         0.000000            0.000000       0.000000          0.000000  2019-12-17T17:52:20+00:00          NaN
2                3            0           1       60.00     60.000000   0.000000   0.000000  0.000000  3.457999      R              1  12/17/2019 09:52:50      0      0      0      0           0.0                  0.0          NaN          NaN  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0         0.000000            0.000000       0.000000          0.000000  2019-12-17T17:52:50+00:00          NaN
3                4            0           1       89.42     89.419998   0.000000   0.000000  0.000000  3.458152      S            192  12/17/2019 09:53:20      0      0      0      0           0.0                  0.0          NaN          NaN  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0         0.000000            0.000000       0.000000          0.000000  2019-12-17T17:53:20+00:00          NaN
4                5            0           1       89.42     89.419998   0.000000   0.000000  0.000000  3.458228      R            192  12/17/2019 11:15:57      0      0      0      0           0.0                  0.0          NaN          NaN  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0         0.000000            0.000000       0.000000          0.000000  2019-12-17T19:15:57+00:00          NaN
...            ...          ...         ...         ...           ...        ...        ...       ...       ...    ...            ...                  ...    ...    ...    ...    ...           ...                  ...          ...          ...  ...    ...    ...    ...    ...    ...    ...    ...    ...     ...     ...     ...     ...     ...     ...              ...                 ...            ...               ...                        ...          ...
546943      546944          246          39  1958303.97  23211.070312   4.459139  16.402617 -0.691691  2.700771      D              5  01/09/2020 03:18:48  64541     22      0      0           0.0                  0.0          NaN          NaN  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0         2.011044            4.459139       8.126739         16.402617  2020-01-09T11:18:48+00:00          NaN
546944      546945          246          39  1958305.13  23212.230469   4.459362  16.403219 -0.691691  2.700008      D            133  01/09/2020 03:18:49  64541     22      0      0           0.0                  0.0          NaN          NaN  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0         2.011044            4.459362       8.126739         16.403219  2020-01-09T11:18:49+00:00          NaN
546945      546946          247          41  1958305.16      0.030000   0.000006   0.000016  1.618448  2.760967      C              0  01/09/2020 03:18:49  64541     22      0      0           0.0                  0.0          NaN          NaN  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0         2.011050            4.459362       8.126755         16.403219  2020-01-09T11:18:49+00:00          NaN
546946      546947          247          41  1958305.32      0.190000   0.000078   0.000215  1.612268  2.777752      C              5  01/09/2020 03:18:49  64541     22      0      0           0.0                  0.0          NaN          NaN  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0         2.011122            4.459362       8.126954         16.403219  2020-01-09T11:18:49+00:00          NaN
546947      546948          247          41  1958305.42      0.290000   0.000122   0.000340  1.612039  2.784771      C              5  01/09/2020 03:18:49  64541     22      0      0           0.0                  0.0          NaN          NaN  ...    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0     0.0     0.0     0.0     0.0     0.0     0.0         2.011167            4.459362       8.127079         16.403219  2020-01-09T11:18:49+00:00          NaN

[546948 rows x 44 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 546948 entries, 0 to 546947
Data columns (total 44 columns):
 #   Column               Non-Null Count   Dtype   
---  ------               --------------   -----   
 0   data_point           546948 non-null  int32   
 1   cycle_index          546948 non-null  int32   
 2   step_index           546948 non-null  int16   
 3   test_time            546948 non-null  float64 
 4   step_time            546948 non-null  float32 
 5   _capacity            546948 non-null  float64 
 6   _energy              546948 non-null  float64 
 7   current              546948 non-null  float32 
 8   voltage              546948 non-null  float32 
 9   _state               546948 non-null  object  
 10  _ending_status       546948 non-null  category
 11  date_time            546948 non-null  object  
 12  loop1                546948 non-null  int64   
 13  loop2                546948 non-null  int64   
 14  loop3                546948 non-null  int64   
 15  loop4                546948 non-null  int64   
 16  ac_impedence         546948 non-null  float32 
 17  internal_resistance  546948 non-null  float32 
 18  _wf_chg_cap          0 non-null       float32 
 19  _wf_dis_cap          0 non-null       float32 
 20  _wf_chg_e            0 non-null       float32 
 21  _wf_dis_e            0 non-null       float32 
 22  _range               546948 non-null  uint8   
 23  _var1                546948 non-null  float16 
 24  _var2                546948 non-null  float16 
 25  _var3                546948 non-null  float16 
 26  _var4                546948 non-null  float16 
 27  _var5                546948 non-null  float16 
 28  _var6                546948 non-null  float16 
 29  _var7                546948 non-null  float16 
 30  _var8                546948 non-null  float16 
 31  _var9                546948 non-null  float16 
 32  _var10               546948 non-null  float16 
 33  _var11               546948 non-null  float16 
 34  _var12               546948 non-null  float16 
 35  _var13               546948 non-null  float16 
 36  _var14               546948 non-null  float16 
 37  _var15               546948 non-null  float16 
 38  charge_capacity      546948 non-null  float64 
 39  discharge_capacity   546948 non-null  float64 
 40  charge_energy        546948 non-null  float64 
 41  discharge_energy     546948 non-null  float64 
 42  date_time_iso        546948 non-null  object  
 43  temperature          0 non-null       float64 
dtypes: category(1), float16(15), float32(9), float64(8), int16(1), int32(2), int64(4), object(3), uint8(1)
memory usage: 103.3+ MB

```




## Inspect Structured Files





## Inspect Feature Matrices

Example:

```shell
S: beep inspect FeatureMatrix-2021-21-09_20.50.32.550211.json.gz 

2021-09-22 15:54:23 INFO     Loaded FeatureMatrix-2021-21-09_20.50.32.550211.json.gz as type <class 'beep.features.base.BEEPFeatureMatrix'>.


BEEP Feature Matrix: FeatureMatrix-2021-21-09_20.50.32.550211.json.gz



Featurizers:

        Featurizer beep.features.core HPPCResistanceVoltageFeatures
        {'@class': 'HPPCResistanceVoltageFeatures',
         '@module': 'beep.features.core',
         'hyperparameters': {'cycle_index_filter': 6,
                             'diag_pos': 1,
                             'parameters_path': '/Users/ardunn/alex/tri/code/beep/beep/protocol_parameters',
                             'soc_window': 8,
                             'test_time_filter_sec': 1000000},
         'linked_datapath_semiunique_id': 'barcode:0000FB-channel:50-protocol:PreDiag_000440.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000440_0000FB_structure.json',
         'metadata': {'barcode': '0000FB',
                      'channel_id': 50,
                      'protocol': 'PreDiag_000440.000'},
         'paths': {'structured': '/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000440_0000FB_structure.json'}}

        Featurizer beep.features.core CycleSummaryStats
        {'@class': 'CycleSummaryStats',
         '@module': 'beep.features.core',
         'hyperparameters': {'cycle_comp_num': [10, 100],
                             'statistics': ['var',
                                            'min',
                                            'mean',
                                            'skew',
                                            'kurtosis',
                                            'abs',
                                            'square']},
         'linked_datapath_semiunique_id': 'barcode:0000FB-channel:50-protocol:PreDiag_000440.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000440_0000FB_structure.json',
         'metadata': {'barcode': '0000FB',
                      'channel_id': 50,
                      'protocol': 'PreDiag_000440.000'},
         'paths': {'structured': '/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000440_0000FB_structure.json'}}

        Featurizer beep.features.core CycleSummaryStats
        {'@class': 'CycleSummaryStats',
         '@module': 'beep.features.core',
         'hyperparameters': {'cycle_comp_num': [11, 101],
                             'statistics': ['var',
                                            'min',
                                            'mean',
                                            'skew',
                                            'kurtosis',
                                            'abs',
                                            'square']},
         'linked_datapath_semiunique_id': 'barcode:0000FB-channel:50-protocol:PreDiag_000440.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000440_0000FB_structure.json',
         'metadata': {'barcode': '0000FB',
                      'channel_id': 50,
                      'protocol': 'PreDiag_000440.000'},
         'paths': {'structured': '/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000440_0000FB_structure.json'}}

        Featurizer beep.features.core HPPCResistanceVoltageFeatures
        {'@class': 'HPPCResistanceVoltageFeatures',
         '@module': 'beep.features.core',
         'hyperparameters': {'cycle_index_filter': 6,
                             'diag_pos': 1,
                             'parameters_path': '/Users/ardunn/alex/tri/code/beep/beep/protocol_parameters',
                             'soc_window': 8,
                             'test_time_filter_sec': 1000000},
         'linked_datapath_semiunique_id': 'barcode:00004C-channel:33-protocol:PredictionDiagnostics_000132.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json',
         'metadata': {'barcode': '00004C',
                      'channel_id': 33,
                      'protocol': 'PredictionDiagnostics_000132.000'},
         'paths': {'structured': '/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json'}}

        Featurizer beep.features.core CycleSummaryStats
        {'@class': 'CycleSummaryStats',
         '@module': 'beep.features.core',
         'hyperparameters': {'cycle_comp_num': [10, 100],
                             'statistics': ['var',
                                            'min',
                                            'mean',
                                            'skew',
                                            'kurtosis',
                                            'abs',
                                            'square']},
         'linked_datapath_semiunique_id': 'barcode:00004C-channel:33-protocol:PredictionDiagnostics_000132.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json',
         'metadata': {'barcode': '00004C',
                      'channel_id': 33,
                      'protocol': 'PredictionDiagnostics_000132.000'},
         'paths': {'structured': '/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json'}}

        Featurizer beep.features.core CycleSummaryStats
        {'@class': 'CycleSummaryStats',
         '@module': 'beep.features.core',
         'hyperparameters': {'cycle_comp_num': [11, 101],
                             'statistics': ['var',
                                            'min',
                                            'mean',
                                            'skew',
                                            'kurtosis',
                                            'abs',
                                            'square']},
         'linked_datapath_semiunique_id': 'barcode:00004C-channel:33-protocol:PredictionDiagnostics_000132.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json',
         'metadata': {'barcode': '00004C',
                      'channel_id': 33,
                      'protocol': 'PredictionDiagnostics_000132.000'},
         'paths': {'structured': '/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json'}}

Matrix:
                                                    D_1::HPPCResistanceVoltageFeatures::6262aa8b2c9ce9530d53f73943e5b465a1946f39be2ad2a3ede05f49e6f9f2d2  ...  var_v_diff::HPPCResistanceVoltageFeatures::6262aa8b2c9ce9530d53f73943e5b465a1946f39be2ad2a3ede05f49e6f9f2d2
filename                                                                                                                                                  ...                                                                                                             
/Users/ardunn/alex/tri/code/beep/beep/tests/tes...                                          -0.075467                                                     ...                                           0.000186                                                          
/Users/ardunn/alex/tri/code/beep/beep/tests/tes...                                          -0.090097                                                     ...                                           0.002462                                                          

[2 rows x 132 columns]
<class 'pandas.core.frame.DataFrame'>
Index: 2 entries, /Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000440_0000FB_structure.json to /Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json
Columns: 132 entries, D_1::HPPCResistanceVoltageFeatures::6262aa8b2c9ce9530d53f73943e5b465a1946f39be2ad2a3ede05f49e6f9f2d2 to var_v_diff::HPPCResistanceVoltageFeatures::6262aa8b2c9ce9530d53f73943e5b465a1946f39be2ad2a3ede05f49e6f9f2d2
dtypes: float64(132)
memory usage: 2.1+ KB
```



