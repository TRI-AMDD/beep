# Inspect


BEEP inspect is a debugging and analysis command which can be used to examine any 
serialized beep object directly from the command line.

The objects that can be inspected are:

- **Raw cycler files** compatible with BEEP, which will be ingested and represented as a `BEEPDatapath`. [Example: Inspect Raw Files](#inspect-raw-files)
- **Structured cycler files** serialized by BEEP to disk as json, represented as a `BEEPDatapath`. [Example: Inspect Structured Files](#inspect-structured-files)
- **Feature matrices** serialized to disk as json. [Example: Inspect Feature Matrices](#inspect-feature-matrices)
- **Individual `BEEPFeaturizer`s** serialized to disk as json. [Example: Inspect Featurizers](#inspect-featurizers)
- **Linear `BEEPLinearModelExperiment`s** serialized to disk as json. [Example: Inspect Models](#inspect-models)

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

2021-09-22 16:01:33 DEBUG    Loaded potential raw file beep/tests/test_files/PreDiag_000287_000128.092 as Datapath.
2021-09-22 16:01:34 INFO     Loaded beep/tests/test_files/PreDiag_000287_000128.092 as type <class 'beep.structure.maccor.MaccorDatapath'>.


BEEP Datapath: beep/tests/test_files/PreDiag_000287_000128.092



Semiunique id: 'barcode:000128-channel:92-protocol:PreDiag_000287.000-schema:beep/validation_schemas/schema-maccor-2170.yaml-structured:False-legacy:False-raw_path:beep/tests/test_files/PreDiag_000287_000128.092-structured_path:None'

File paths
{'metadata': 'beep/tests/test_files/PreDiag_000287_000128.092',
 'raw': 'beep/tests/test_files/PreDiag_000287_000128.092'}

File metadata:
{'_today_datetime': '12/17/2019',
 'barcode': '000128',
 'channel_id': 92,
 'filename': 'C:\\Users\\Maccor Tester User\\Documents\\Backup\\STANFORD '
             'LOANER #1\\STANFORD LOANER #1\\PreDiag_000287_000128.092',
 'protocol': 'PreDiag_000287.000',
 'start_datetime': '12/17/2019'}

Validation schema: beep/validation_schemas/schema-maccor-2170.yaml

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

Example:

```shell
$: beep inspect 2017-12-04_4_65C-69per_6C_CH29_structured_new.json.gz

2021-09-22 16:04:01 INFO     Loaded beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29_structured_new.json.gz as type <class 'beep.structure.arbin.ArbinDatapath'>.


BEEP Datapath: beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29_structured_new.json.gz



Semiunique id: 'barcode:EL151000429559-channel:28-protocol:2017-12-04_tests\20170630-4_65C_69per_6C.sdu-schema:beep/validation_schemas/schema-arbin-lfp.yaml-structured:True-legacy:True-raw_path:beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv-structured_path:None'

File paths
{'metadata': 'beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29_Metadata.csv',
 'raw': 'beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv'}

File metadata:
{'barcode': 'EL151000429559',
 'channel_id': 28,
 'protocol': '2017-12-04_tests\\20170630-4_65C_69per_6C.sdu'}

Validation schema: beep/validation_schemas/schema-arbin-lfp.yaml

Structuring parameters:
{'charge_axis': 'charge_capacity',
 'diagnostic_available': False,
 'diagnostic_resolution': 500,
 'discharge_axis': 'voltage',
 'full_fast_charge': 0.8,
 'nominal_capacity': 1.1,
 'resolution': 1000,
 'v_range': None}

Structured attributes:

structured_summary:
     cycle_index  discharge_capacity  charge_capacity  discharge_energy  charge_energy  dc_internal_resistance  temperature_maximum  temperature_average  temperature_minimum              date_time_iso  energy_efficiency  charge_throughput  energy_throughput  charge_duration  time_temperature_integrated  paused       CV_time  CV_current
0              0            1.940235         1.432850          6.142979       4.725729                0.029954            34.222515            32.666893            20.699526  2017-12-05T03:37:36+00:00           1.299901           1.432850           4.725729          32768.0                 48977.078333   13312  50158.164062    0.000029
1              1            1.060343         1.061786          3.219735       3.703581                0.017906            35.375809            32.387295            30.235437  2017-12-06T04:33:04+00:00           0.869357           2.494636           8.429310            640.0                  1927.509119       0   1577.987427    0.062837
2              2            1.065412         1.065450          3.235807       3.708392                0.017649            35.384602            32.472481            30.254265  2017-12-06T05:32:48+00:00           0.872563           3.560086          12.137702            640.0                  1931.514632       0   1570.970459    0.046215
3              3            1.066605         1.066726          3.238866       3.711425                0.017506            35.265358            32.420013            30.159765  2017-12-06T06:32:32+00:00           0.872675           4.626812          15.849127            640.0                  1995.445402       0   1552.032471    0.045087
4              4            1.066988         1.067148          3.239955       3.712645                0.017409            35.280449            32.407478            30.157305  2017-12-06T07:34:24+00:00           0.872681           5.693960          19.561773            512.0                  1926.752726       0   1548.414551    0.052290
..           ...                 ...              ...               ...            ...                     ...                  ...                  ...                  ...                        ...                ...                ...                ...              ...                          ...     ...           ...         ...
183          183            1.033595         1.033812          3.046869       3.615792                0.016889            37.566158            32.988796            30.226278  2017-12-13T19:12:00+00:00           0.842656         194.065491         676.183533            640.0                  2026.858687       0   1590.061523    0.034429
184          184            1.033454         1.033584          3.042845       3.613951                0.016827            37.129795            32.981796            30.181578  2017-12-13T20:13:52+00:00           0.841972         195.099075         679.797485            640.0                  1955.320325       0   1589.661377    0.031118
185          185            1.032677         1.032898          3.040163       3.612450                0.016875            37.126766            32.851368            30.145836  2017-12-13T21:13:36+00:00           0.841579         196.131973         683.409912            640.0                  1950.674312       0   1530.225342    0.021825
186          186            1.032823         1.033198          3.041561       3.613732                0.016875            37.236954            32.925690            30.300278  2017-12-13T22:13:20+00:00           0.841668         197.165176         687.023682            640.0                  1954.338322       0   1590.264771    0.026628
187          187            1.032616         1.032862          3.039321       3.612212                0.016840            37.159687            32.952461            30.114653  2017-12-13T23:13:04+00:00           0.841402         198.198029         690.635864            640.0                  1955.141357       0   1590.173706    0.025024

[188 rows x 18 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 188 entries, 0 to 187
Data columns (total 18 columns):
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   cycle_index                  188 non-null    int32  
 1   discharge_capacity           188 non-null    float64
 2   charge_capacity              188 non-null    float64
 3   discharge_energy             188 non-null    float64
 4   charge_energy                188 non-null    float64
 5   dc_internal_resistance       188 non-null    float32
 6   temperature_maximum          188 non-null    float32
 7   temperature_average          188 non-null    float32
 8   temperature_minimum          188 non-null    float32
 9   date_time_iso                188 non-null    object 
 10  energy_efficiency            188 non-null    float32
 11  charge_throughput            188 non-null    float32
 12  energy_throughput            188 non-null    float32
 13  charge_duration              188 non-null    float32
 14  time_temperature_integrated  188 non-null    float64
 15  paused                       188 non-null    int32  
 16  CV_time                      188 non-null    float32
 17  CV_current                   188 non-null    float32
dtypes: float32(10), float64(5), int32(2), object(1)
memory usage: 17.8+ KB
None

structured_data:
         voltage     test_time   current  charge_capacity  discharge_capacity  charge_energy  discharge_energy  internal_resistance  temperature  cycle_index  step_type
0       2.800000  88438.740972 -3.070090         1.319212            1.788713       4.354456          5.709161             0.028598    31.529890            0  discharge
1       2.800701  85441.894275 -4.256237         1.370451            1.831461       4.519928          5.826991             0.029763    32.309685            0  discharge
2       2.801401  58527.144191 -3.221379         0.921136            1.386086       3.038029          4.406347             0.028391    32.847729            0  discharge
3       2.802102  31612.394108 -2.186522         0.471821            0.940710       1.556129          2.985704             0.027020    33.385773            0  discharge
4       2.802803   4697.644024 -1.151665         0.022506            0.495335       0.074230          1.565060             0.025648    33.923817            0  discharge
...          ...           ...       ...              ...                 ...            ...               ...                  ...          ...          ...        ...
375995       NaN           NaN       NaN         1.427113                 NaN            NaN               NaN                  NaN          NaN          187     charge
375996       NaN           NaN       NaN         1.428547                 NaN            NaN               NaN                  NaN          NaN          187     charge
375997       NaN           NaN       NaN         1.429981                 NaN            NaN               NaN                  NaN          NaN          187     charge
375998       NaN           NaN       NaN         1.431416                 NaN            NaN               NaN                  NaN          NaN          187     charge
375999       NaN           NaN       NaN         1.432850                 NaN            NaN               NaN                  NaN          NaN          187     charge

[376000 rows x 11 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 376000 entries, 0 to 375999
Data columns (total 11 columns):
 #   Column               Non-Null Count   Dtype   
---  ------               --------------   -----   
 0   voltage              325974 non-null  float32 
 1   test_time            325974 non-null  float64 
 2   current              325974 non-null  float32 
 3   charge_capacity      376000 non-null  float32 
 4   discharge_capacity   325974 non-null  float32 
 5   charge_energy        325974 non-null  float32 
 6   discharge_energy     325974 non-null  float32 
 7   internal_resistance  325974 non-null  float32 
 8   temperature          325974 non-null  float32 
 9   cycle_index          376000 non-null  int32   
 10  step_type            376000 non-null  category
dtypes: category(1), float32(8), float64(1), int32(1)
memory usage: 16.1 MB
None

diagnostic_data:
        No object.

diagnostic_summary:
        No object.

raw_data:
        No object.

```





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
                             'parameters_path': 'beep/protocol_parameters',
                             'soc_window': 8,
                             'test_time_filter_sec': 1000000},
         'linked_datapath_semiunique_id': 'barcode:0000FB-channel:50-protocol:PreDiag_000440.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:beep/tests/test_files/PreDiag_000440_0000FB_structure.json',
         'metadata': {'barcode': '0000FB',
                      'channel_id': 50,
                      'protocol': 'PreDiag_000440.000'},
         'paths': {'structured': 'beep/tests/test_files/PreDiag_000440_0000FB_structure.json'}}

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
         'linked_datapath_semiunique_id': 'barcode:0000FB-channel:50-protocol:PreDiag_000440.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:beep/tests/test_files/PreDiag_000440_0000FB_structure.json',
         'metadata': {'barcode': '0000FB',
                      'channel_id': 50,
                      'protocol': 'PreDiag_000440.000'},
         'paths': {'structured': 'beep/tests/test_files/PreDiag_000440_0000FB_structure.json'}}

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
         'linked_datapath_semiunique_id': 'barcode:0000FB-channel:50-protocol:PreDiag_000440.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:beep/tests/test_files/PreDiag_000440_0000FB_structure.json',
         'metadata': {'barcode': '0000FB',
                      'channel_id': 50,
                      'protocol': 'PreDiag_000440.000'},
         'paths': {'structured': 'beep/tests/test_files/PreDiag_000440_0000FB_structure.json'}}

        Featurizer beep.features.core HPPCResistanceVoltageFeatures
        {'@class': 'HPPCResistanceVoltageFeatures',
         '@module': 'beep.features.core',
         'hyperparameters': {'cycle_index_filter': 6,
                             'diag_pos': 1,
                             'parameters_path': 'beep/protocol_parameters',
                             'soc_window': 8,
                             'test_time_filter_sec': 1000000},
         'linked_datapath_semiunique_id': 'barcode:00004C-channel:33-protocol:PredictionDiagnostics_000132.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json',
         'metadata': {'barcode': '00004C',
                      'channel_id': 33,
                      'protocol': 'PredictionDiagnostics_000132.000'},
         'paths': {'structured': 'beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json'}}

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
         'linked_datapath_semiunique_id': 'barcode:00004C-channel:33-protocol:PredictionDiagnostics_000132.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json',
         'metadata': {'barcode': '00004C',
                      'channel_id': 33,
                      'protocol': 'PredictionDiagnostics_000132.000'},
         'paths': {'structured': 'beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json'}}

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
         'linked_datapath_semiunique_id': 'barcode:00004C-channel:33-protocol:PredictionDiagnostics_000132.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json',
         'metadata': {'barcode': '00004C',
                      'channel_id': 33,
                      'protocol': 'PredictionDiagnostics_000132.000'},
         'paths': {'structured': 'beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json'}}

Matrix:
                                                    D_1::HPPCResistanceVoltageFeatures::6262aa8b2c9ce9530d53f73943e5b465a1946f39be2ad2a3ede05f49e6f9f2d2  ...  var_v_diff::HPPCResistanceVoltageFeatures::6262aa8b2c9ce9530d53f73943e5b465a1946f39be2ad2a3ede05f49e6f9f2d2
filename                                                                                                                                                  ...                                                                                                             
beep/tests/tes...                                          -0.075467                                                     ...                                           0.000186                                                          
beep/tests/tes...                                          -0.090097                                                     ...                                           0.002462                                                          

[2 rows x 132 columns]
<class 'pandas.core.frame.DataFrame'>
Index: 2 entries, beep/tests/test_files/PreDiag_000440_0000FB_structure.json to beep/tests/test_files/PredictionDiagnostics_000132_00004C_structure.json
Columns: 132 entries, D_1::HPPCResistanceVoltageFeatures::6262aa8b2c9ce9530d53f73943e5b465a1946f39be2ad2a3ede05f49e6f9f2d2 to var_v_diff::HPPCResistanceVoltageFeatures::6262aa8b2c9ce9530d53f73943e5b465a1946f39be2ad2a3ede05f49e6f9f2d2
dtypes: float64(132)
memory usage: 2.1+ KB
```


## Inspect Featurizers

Example:

```shell
$: beep inspect HPPCFeaturizer.json.gz 
2021-09-22 16:06:42 INFO     Loaded beep/tests/test_files/modelling_test_files/HPPCFeaturizer.json.gz as type <class 'beep.features.core.HPPCResistanceVoltageFeatures'>.


BEEP Featurizer: beep/tests/test_files/modelling_test_files/HPPCFeaturizer.json.gz



File paths:
{'structured': 'beep/CLI_TEST_FILES_FEATURIZATION/PreDiag_000440_0000FB_structure.json'}

Linked datapath semiunique id: barcode:0000FB-channel:50-protocol:PreDiag_000440.000-schema:None-structured:True-legacy:True-raw_path:None-structured_path:beep/CLI_TEST_FILES_FEATURIZATION/PreDiag_000440_0000FB_structure.json

Hyperparameters:
{'cycle_index_filter': 6,
 'diag_pos': 1,
 'parameters_path': 'beep/protocol_parameters',
 'soc_window': 8,
 'test_time_filter_sec': 1000000}

Metadata:
{'barcode': '0000FB', 'channel_id': 50, 'protocol': 'PreDiag_000440.000'}

Features:
   r_c_0s_00  r_c_0s_10  r_c_0s_20  r_c_0s_30  r_c_0s_40  r_c_0s_50  r_c_0s_60  r_c_0s_70  r_c_0s_80  r_c_3s_00  r_c_3s_10  r_c_3s_20  r_c_3s_30  r_c_3s_40  r_c_3s_50  r_c_3s_60  r_c_3s_70  r_c_3s_80  r_c_end_00  ...  skew_ocv  kurtosis_ocv   sum_ocv  sum_square_ocv  var_v_diff  min_v_diff  mean_v_diff  skew_v_diff  kurtosis_v_diff  sum_v_diff  sum_square_v_diff       D_1       D_2       D_3       D_4       D_5       D_6       D_7       D_8
0  -0.056034  -0.063766   -0.07963  -0.105001  -0.091609  -0.095464  -0.073553   -0.06692  -0.064657  -0.037199  -0.071951  -0.077876  -0.128588  -0.103652  -0.106871  -0.096638  -0.066802  -0.074038   -0.053153  ...  1.674431      7.472183  0.045535        0.000641    0.000186    -0.00181     0.012954     0.887649         2.940287    14.16811           0.373482 -0.075467 -0.097516 -0.230871 -0.163967 -0.158305 -0.137443  0.070989  0.098653

[1 rows x 76 columns]

```





## Inspect Models

Example:

```shell
$: beep inspect model-src.json.gz 

2021-09-22 16:06:04 WARNING  Number of samples (4) less than number of features (179); may cause overfitting.
2021-09-22 16:06:04 INFO     Loaded beep/tests/test_files/modelling_test_files/model-src.json.gz as type <class 'beep.model.BEEPLinearModelExperiment'>.


BEEP Linear Model Experiment: beep/tests/test_files/modelling_test_files/model-src.json.gz



Targets: ['capacity_0.92::TrajectoryFastCharge']

Model name: lasso

Impute strategy: median

Homogenize features: True

NaN Thresholds:
        -train_feature_drop_nan_thresh: 0.95
        -train_sample_drop_nan_thresh: 0.5
        -predict_sample_nan_thresh: 0.75

Model parameters:
        - coef_: [0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0]
        - intercept_: 113.25
        - optimal_hyperparameters: {'alpha': 98.35818271439722}

Matrices:

feature_matrix
                                                    D_1::HPPCResistanceVoltageFeatures::6262aa8b2c9ce9530d53f73943e5b465a1946f39be2ad2a3ede05f49e6f9f2d2  ...  var_v_diff::HPPCResistanceVoltageFeatures::6262aa8b2c9ce9530d53f73943e5b465a1946f39be2ad2a3ede05f49e6f9f2d2
filename                                                                                                                                                  ...                                                                                                             
beep/CLI_TEST_...                                          -0.075467                                                     ...                                           0.000186                                                          
beep/CLI_TEST_...                                          -0.090097                                                     ...                                           0.002462                                                          
beep/CLI_TEST_...                                          -0.145030                                                     ...                                           0.002416                                                          
beep/CLI_TEST_...                                          -0.052108                                                     ...                                           0.000848                                                          

[4 rows x 179 columns]
<class 'pandas.core.frame.DataFrame'>
Index: 4 entries, beep/CLI_TEST_FILES_FEATURIZATION/PreDiag_000440_0000FB_structure.json to beep/CLI_TEST_FILES_FEATURIZATION/PredictionDiagnostics_000136_00002D_structure.json
Columns: 179 entries, D_1::HPPCResistanceVoltageFeatures::6262aa8b2c9ce9530d53f73943e5b465a1946f39be2ad2a3ede05f49e6f9f2d2 to var_v_diff::HPPCResistanceVoltageFeatures::6262aa8b2c9ce9530d53f73943e5b465a1946f39be2ad2a3ede05f49e6f9f2d2
dtypes: float64(179)
memory usage: 5.6+ KB
None

target_matrix
                                                    capacity_0.83::TrajectoryFastCharge::319cec55cc030c1911b2530cae3fc2df8d3c24912ae01ee4172ea4ca4caddec8  ...  rpt_1Cdischarge_energy0.8_real_regular_throughput::DiagnosticProperties::9fb32356773f0c4f8c27fc9528ca4a986dc928fbadbd859b67a8892e7daac72e
filename                                                                                                                                                   ...                                                                                                                                           
beep/CLI_TEST_...                                                284                                                      ...                                                NaN                                                                                        
beep/CLI_TEST_...                                                 58                                                      ...                                        1266.108637                                                                                        
beep/CLI_TEST_...                                                 85                                                      ...                                                NaN                                                                                        
beep/CLI_TEST_...                                                101                                                      ...                                                NaN                                                                                        

[4 rows x 11 columns]
<class 'pandas.core.frame.DataFrame'>
Index: 4 entries, beep/CLI_TEST_FILES_FEATURIZATION/PreDiag_000440_0000FB_structure.json to beep/CLI_TEST_FILES_FEATURIZATION/PredictionDiagnostics_000136_00002D_structure.json
Data columns (total 11 columns):
 #   Column                                                                                                                                           Non-Null Count  Dtype  
---  ------                                                                                                                                           --------------  -----  
 0   capacity_0.83::TrajectoryFastCharge::319cec55cc030c1911b2530cae3fc2df8d3c24912ae01ee4172ea4ca4caddec8                                            4 non-null      int64  
 1   capacity_0.86::TrajectoryFastCharge::319cec55cc030c1911b2530cae3fc2df8d3c24912ae01ee4172ea4ca4caddec8                                            4 non-null      int64  
 2   capacity_0.89::TrajectoryFastCharge::319cec55cc030c1911b2530cae3fc2df8d3c24912ae01ee4172ea4ca4caddec8                                            4 non-null      int64  
 3   capacity_0.8::TrajectoryFastCharge::319cec55cc030c1911b2530cae3fc2df8d3c24912ae01ee4172ea4ca4caddec8                                             4 non-null      int64  
 4   capacity_0.92::TrajectoryFastCharge::319cec55cc030c1911b2530cae3fc2df8d3c24912ae01ee4172ea4ca4caddec8                                            4 non-null      int64  
 5   capacity_0.95::TrajectoryFastCharge::319cec55cc030c1911b2530cae3fc2df8d3c24912ae01ee4172ea4ca4caddec8                                            4 non-null      int64  
 6   capacity_0.98::TrajectoryFastCharge::319cec55cc030c1911b2530cae3fc2df8d3c24912ae01ee4172ea4ca4caddec8                                            4 non-null      int64  
 7   initial_regular_throughput::DiagnosticProperties::9fb32356773f0c4f8c27fc9528ca4a986dc928fbadbd859b67a8892e7daac72e                               1 non-null      float64
 8   rpt_1Cdischarge_energy0.8_cycle_index::DiagnosticProperties::9fb32356773f0c4f8c27fc9528ca4a986dc928fbadbd859b67a8892e7daac72e                    1 non-null      float64
 9   rpt_1Cdischarge_energy0.8_normalized_regular_throughput::DiagnosticProperties::9fb32356773f0c4f8c27fc9528ca4a986dc928fbadbd859b67a8892e7daac72e  1 non-null      float64
 10  rpt_1Cdischarge_energy0.8_real_regular_throughput::DiagnosticProperties::9fb32356773f0c4f8c27fc9528ca4a986dc928fbadbd859b67a8892e7daac72e        1 non-null      float64
dtypes: float64(4), int64(7)
memory usage: 556.0+ bytes

```