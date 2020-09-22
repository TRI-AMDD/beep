import unittest
import os
from beep.protocol.biologic_mb_to_maccor import BiologicMbToMaccorProcedure

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")
SAMPLE_MB_FILE_NAME = "BCS - 171.64.160.115_Ta19_ourprotocol_gdocSEP2019_CC7.mps"
CONVERTED_OUTPUT_FILE_NAME = "test_biologic_mb_to_maccor_output_diagnostic"

SAMPLE_MB_TEXT = """BT-LAB SETTING FILE

Number of linked techniques : 1

Filename : C:\\Users\\sample.mps

Device : BCS-805
Ecell ctrl range : min = 0.00 V, max = 10.00 V
Electrode material : 
Initial state : 
Electrolyte : 
Comments : 
Mass of active material : 0.001 mg
at x = 0.000
Molecular weight of active material (at x = 0) : 0.001 g/mol
Atomic weight of intercalated ion : 0.001 g/mol
Acquisition started at : xo = 0.000
Number of e- transfered per intercalated ion : 1
for DX = 1, DQ = 26.802 mA.h
Battery capacity : 2.280 mA.h
Electrode surface area : 0.001 cm
Characteristic mass : 9.130 mg
Cycle Definition : Charge/Discharge alternance
Turn to OCV between techniques

Technique : 1
Modulo Bat
Ns                  0                   1                   2                   3                   
ctrl_type           CV                  Rest                Loop                CR                  
Apply I/C           C / N               I                   C / N               C / N               
ctrl1_val           1.500                                                       1.500               
ctrl1_val_unit      V                                                           Ohm                 
ctrl1_val_vs        Ref                                                         <None>               
ctrl2_val                                                                                           
ctrl2_val_unit                                                                                      
ctrl2_val_vs                                                                                        
ctrl3_val           1.700               1.700               1.700               1.700               
ctrl3_val_unit      1.700               1.700               1.700               1.700               
ctrl3_val_vs        1.700               1.700               1.700               1.700               
N                   0.00                0.01                0.01                0.01                
charge/discharge    Charge              Charge              Charge              Charge              
ctrl_seq            0                   0                   0                   0                   
ctrl_repeat         0                   0                   5                   5                   
ctrl_trigger        Rising Edge         Rising Edge         Rising Edge         Rising Edge         
ctrl_TO_t           0.000               0.000               0.000               0.000               
ctrl_TO_t_unit      s                   s                   s                   s                   
ctrl_Nd             6                   6                   6                   6                   
ctrl_Na             2                   2                   2                   2                   
ctrl_corr           0                   0                   0                   0                   
lim_nb              2                   2                   0                   1                   
lim1_type           Time                Time                Time                Ecell               
lim1_comp           >                   >                   >                   >                   
lim1_Q              Q limit             Q limit             Q limit             Q limit             
lim1_value          10.000              1.000               1.000               1.000               
lim1_value_unit     s                   h                   h                   V                   
lim1_action         Next sequence       Next sequence       Next sequence       Next sequence       
lim1_seq            1                   2                   3                   4                   
lim2_type           I                   I                   I                   I                   
lim2_comp           <                   <                   <                   <                   
lim2_Q              Q limit             Q limit             Q limit             Q limit             
lim2_value          4.000               4.000               4.000               4.000               
lim2_value_unit     mA                  mA                  mA                  mA                  
lim2_action         Goto sequence       Goto sequence       Next sequence       Next sequence       
lim2_seq            3                   1                   3                   3                   
lim3_type           Time                Time                Time                Time                
lim3_comp           <                   <                   <                   <                   
lim3_Q              Q limit             Q limit             Q limit             Q limit             
lim3_value          0.000               0.000               0.000               0.000               
lim3_value_unit     s                   s                   s                   s                   
lim3_action         Next sequence       Next sequence       Next sequence       Next sequence       
lim3_seq            1                   1                   1                   1                   
rec_nb              1                   1                   0                   1                   
rec1_type           I                   Power               Power               Power               
rec1_value          5.000               5.000               5.000               5.000               
rec1_value_unit     µA                  W                   W                   W                   
rec2_type           Time                Time                Time                Time                
rec2_value          0.000               0.000               0.000               0.000               
rec2_value_unit     s                   s                   s                   s                   
rec3_type           Time                Time                Time                Time                
rec3_value          0.000               0.000               0.000               0.000               
rec3_value_unit     s                   s                   s                   s                   
E range min (V)     -10.000             -10.000             -10.000             -10.000             
E range max (V)     10.000              10.000              10.000              10.000              
I Range             100 mA              1 A                 1 A                 100 mA              
I Range min         Unset               Unset               Unset               Unset               
I Range max         Unset               Unset               Unset               Unset               
I Range init        Unset               Unset               Unset               Unset               
auto rest           0                   0                   0                   0                   
Bandwidth           5                   5                   5                   5                   
"""

EXPECTED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<?maccor-application progid="Maccor Procedure File"?>
<MaccorTestProcedure>
  <header>
    <BuildTestVersion>
      <major>1</major>
      <minor>5</minor>
      <release>7006</release>
      <build>32043</build>
    </BuildTestVersion>
    <FileFormatVersion>
      <BTVersion>11</BTVersion>
    </FileFormatVersion>
    <ProcDesc>
      <desc></desc>
    </ProcDesc>
  </header>
  <ProcSteps>
    <TestStep>
      <StepType>  Do 1  </StepType>
      <StepMode>        </StepMode>
      <StepValue></StepValue>
      <Limits/>
      <Ends/>
      <Reports/>
      <Range></Range>
      <Option1></Option1>
      <Option2></Option2>
      <Option3></Option3>
      <StepNote></StepNote>
    </TestStep>
    <TestStep>
      <StepType>Dischrge </StepType>
      <StepMode>Voltage </StepMode>
      <StepValue>1.500</StepValue>
      <Limits/>
      <Ends>
        <EndEntry>
          <EndType>StepTime</EndType>
          <SpecialType> </SpecialType>
          <Oper> = </Oper>
          <Step>003</Step>
          <Value>00:00:10.0</Value>
        </EndEntry>
        <EndEntry>
          <EndType>Current </EndType>
          <SpecialType> </SpecialType>
          <Oper>&lt;= </Oper>
          <Step>006</Step>
          <Value>4.0E-3</Value>
        </EndEntry>
      </Ends>
      <Reports>
        <ReportEntry>
          <ReportType>Current </ReportType>
          <Value>5.0E-6</Value>
        </ReportEntry>
      </Reports>
      <Range>4</Range>
      <Option1>N</Option1>
      <Option2>N</Option2>
      <Option3>N</Option3>
      <StepNote></StepNote>
    </TestStep>
    <TestStep>
      <StepType>  Rest  </StepType>
      <StepMode>        </StepMode>
      <StepValue></StepValue>
      <Limits/>
      <Ends>
        <EndEntry>
          <EndType>StepTime</EndType>
          <SpecialType> </SpecialType>
          <Oper> = </Oper>
          <Step>004</Step>
          <Value>1.0:00:00</Value>
        </EndEntry>
        <EndEntry>
          <EndType>Current </EndType>
          <SpecialType> </SpecialType>
          <Oper>&lt;= </Oper>
          <Step>003</Step>
          <Value>4.0E-3</Value>
        </EndEntry>
      </Ends>
      <Reports>
        <ReportEntry>
          <ReportType>Power </ReportType>
          <Value>5.0</Value>
        </ReportEntry>
      </Reports>
      <Range>4</Range>
      <Option1>N</Option1>
      <Option2>N</Option2>
      <Option3>N</Option3>
      <StepNote></StepNote>
    </TestStep>
    <TestStep>
      <StepType>AdvCycle</StepType>
      <StepMode>        </StepMode>
      <StepValue></StepValue>
      <Limits/>
      <Ends/>
      <Reports/>
      <Range></Range>
      <Option1></Option1>
      <Option2></Option2>
      <Option3></Option3>
      <StepNote></StepNote>
    </TestStep>
    <TestStep>
      <StepType> Loop 1 </StepType>
      <StepMode>        </StepMode>
      <StepValue></StepValue>
      <Limits/>
      <Ends>
        <EndEntry>
          <EndType>Loop Cnt</EndType>
          <SpecialType> </SpecialType>
          <Oper> = </Oper>
          <Step>002</Step>
          <Value></Value>
          <StepValue>5</StepValue>
        </EndEntry>
      </Ends>
      <Reports/>
      <Range></Range>
      <Option1></Option1>
      <Option2></Option2>
      <Option3></Option3>
      <StepNote></StepNote>
    </TestStep>
    <TestStep>
      <StepType>Dischrge </StepType>
      <StepMode>Resistance </StepMode>
      <StepValue>1.500</StepValue>
      <Limits/>
      <Ends>
        <EndEntry>
          <EndType>Voltage </EndType>
          <SpecialType> </SpecialType>
          <Oper>&gt;= </Oper>
          <Step>007</Step>
          <Value>1.0</Value>
        </EndEntry>
      </Ends>
      <Reports>
        <ReportEntry>
          <ReportType>Power </ReportType>
          <Value>5.0</Value>
        </ReportEntry>
      </Reports>
      <Range>4</Range>
      <Option1>N</Option1>
      <Option2>N</Option2>
      <Option3>N</Option3>
      <StepNote></StepNote>
    </TestStep>
    <TestStep>
      <StepType>  End   </StepType>
      <StepMode>        </StepMode>
      <StepValue></StepValue>
      <Limits/>
      <Ends/>
      <Reports/>
      <Range></Range>
      <Option1></Option1>
      <Option2></Option2>
      <Option3></Option3>
      <StepNote></StepNote>
    </TestStep>
  </ProcSteps>
"""


class BiologicMbToMaccorTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_convert_resistance(self):
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_resistance(
                "1.32", "MOhm", "lim3_unit", 1
            ),
            "1.32E6",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_resistance(
                "1.32", "kOhm", "lim3_unit", 1
            ),
            "1.32E3",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_resistance(
                "1.32", "Ohm", "lim3_unit", 1
            ),
            "1.32",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_resistance(
                "1.32", "mOhm", "lim3_unit", 1
            ),
            "1.32E-3",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_resistance(
                "1.32", "\N{Greek Small Letter Mu}Ohm", "lim3_unit", 1
            ),
            "1.32E-6",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_resistance(
                "1.32", "\N{Micro Sign}Ohm", "lim3_unit", 1
            ),
            "1.32E-6",
        )
        self.assertRaises(
            Exception,
            BiologicMbToMaccorProcedure._convert_resistance,
            "4.57",
            "mV",
            "rec2_unit",
            1,
        )

    def test_convert_voltage(self):
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_voltage("4.57", "V", "rec2_unit", 2),
            "4.57",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_voltage("4.57", "mV", "rec2_unit", 2),
            "4.57E-3",
        )
        # wrong unit
        self.assertRaises(
            Exception,
            BiologicMbToMaccorProcedure._convert_voltage,
            "4.57",
            "mA",
            "rec2_unit",
            2,
        )

    def test_convert_current(self):
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_current("2.11", "A", "lim3_unit", 1),
            "2.11",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_current("2.11", "mA", "lim3_unit", 1),
            "2.11E-3",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_current(
                "2.11", "\N{Greek Small Letter Mu}A", "lim3_unit", 1
            ),
            "2.11E-6",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_current(
                "2.11", "\N{Micro Sign}A", "lim3_unit", 1
            ),
            "2.11E-6",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_current("2.11", "nA", "lim3_unit", 1),
            "2.11E-9",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_current("2.11", "pA", "lim3_unit", 1),
            "2.11E-12",
        )
        self.assertRaises(
            Exception,
            BiologicMbToMaccorProcedure._convert_current,
            "4.57",
            "mV",
            "rec2_unit",
            1,
        )

    def test_convert_power(self):
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_power("0.560", "W", "rec1_unit", 1),
            "0.560",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_power("0.560", "mW", "rec1_unit", 1),
            "0.560E-3",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_power(
                "0.560", "\N{Greek Small Letter Mu}W", "rec1_unit", 1
            ),
            "0.560E-6",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_power(
                "0.560", "\N{Micro Sign}W", "rec1_unit", 1
            ),
            "0.560E-6",
        )

    def test_convert_time(self):
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_time("1", "h", "lim3_unit", 1),
            "1:00:00",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_time("56", "mn", "lim3_unit", 1),
            "00:56:00",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_time("1.5", "s", "lim3_unit", 1),
            "00:00:1.5",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_time("10", "ms", "lim3_unit", 1),
            "00:00:0.01",
        )

        # too much specificity for maccor to handle
        self.assertRaises(
            Exception,
            BiologicMbToMaccorProcedure._convert_time,
            "1",
            "ms",
            "lim3_unit",
            1,
        )

        self.assertRaises(
            Exception,
            BiologicMbToMaccorProcedure._convert_time,
            "4.57",
            "mV",
            "rec2_unit",
            1,
        )

    def _test_step_helper(self, expected_step, actual_step):
        self.assertEqual(expected_step["StepMode"], actual_step["StepMode"])
        self.assertEqual(expected_step["StepValue"], actual_step["StepValue"])
        self.assertEqual(expected_step["Range"], actual_step["Range"])
        self.assertEqual(expected_step["Option1"], actual_step["Option1"])
        self.assertEqual(expected_step["Option2"], actual_step["Option2"])
        self.assertEqual(expected_step["Option3"], actual_step["Option3"])
        self.assertEqual(expected_step["StepNote"], actual_step["StepNote"])
        # self.assertEqual(actual_step["Reports"]["ReportEntry"][0], "a")
        if type(expected_step["Reports"]) == str:
            self.assertEqual(expected_step["Reports"], actual_step["Reports"])
        else:
            for report_num, report in enumerate(
                expected_step["Reports"]["ReportEntry"]
            ):
                actual_entries = actual_step["Reports"]["ReportEntry"]
                for key, value in report.items():
                    self.assertEqual(
                        value,
                        actual_entries[report_num][key],
                        msg="bad ReportEntry Field: <{}>, Value:{}".format(key, value),
                    )

        if type(expected_step["Ends"]) == str:
            self.assertEqual(expected_step["Ends"], actual_step["Ends"])
        else:
            for end_num, end in enumerate(expected_step["Ends"]["EndEntry"]):
                actual_end_entries = actual_step["Ends"]["EndEntry"]
                for key, value in end.items():
                    self.assertEqual(
                        value,
                        actual_end_entries[end_num][key],
                        msg="bad ReportEntry Field: <{}>, Value:{}".format(key, value),
                    )

    def test_convert_step_rest(self):
        seq = {
            "ctrl_type": "Rest",
            "ctrl1_val": "",
            "ctrl1_val_unit": "",
            "ctrl1_val_vs": "",
            "Ns": "1",
            "Apply I/C": "I",
            "charge/discharge": "Discharge",
            "lim_nb": "1",
            "lim1_type": "Ecell",
            "lim1_comp": ">",
            "lim1_value": "4.4",
            "lim1_value_unit": "V",
            "lim1_action": "Goto sequence",
            "lim1_seq": "3",
            "rec_nb": "1",
            "rec1_type": "I",
            "rec1_value": "2.2",
            "rec1_value_unit": "A",
        }
        seq_num_by_step_num = {1: 1, 3: 5}
        seq_num_is_non_empty_loop_start = set()
        end_step_num = 5

        rest_step = BiologicMbToMaccorProcedure._create_step(
            seq, seq_num_by_step_num, seq_num_is_non_empty_loop_start, end_step_num
        )
        expected_rest_step = {
            "StepType": "  Rest  ",
            "StepMode": "        ",
            "StepValue": "",
            "Limits": "",
            "Ends": {
                "EndEntry": [
                    {
                        "EndType": "Voltage ",
                        "SpecialType": " ",
                        "Oper": ">= ",
                        "Step": "005",
                        "Value": "4.4",
                    }
                ],
            },
            "Reports": {"ReportEntry": [{"ReportType": "Current ", "Value": "2.2",}],},
            "Range": "4",
            "Option1": "N",
            "Option2": "N",
            "Option3": "N",
            "StepNote": "",
        }

        self._test_step_helper(expected_rest_step, rest_step)

    def test_convert_constant_current_step(self):
        seq = {
            "ctrl_type": "CC",
            "ctrl1_val": "100.00",
            "ctrl1_val_unit": "µA",
            "ctrl1_val_vs": "",
            "Ns": "1",
            "Apply I/C": "I",
            "charge/discharge": "Discharge",
            "lim_nb": "1",
            "lim1_type": "Ecell",
            "lim1_comp": ">",
            "lim1_value": "4.4",
            "lim1_value_unit": "V",
            "lim1_action": "End",
            "lim1_seq": "3",
            "rec_nb": "1",
            "rec1_type": "I",
            "rec1_value": "2.2",
            "rec1_value_unit": "A",
        }
        seq_num_by_step_num = {1: 1, 3: 5}
        seq_num_is_non_empty_loop_start = set()
        end_step_num = 5

        constant_current_step = BiologicMbToMaccorProcedure._create_step(
            seq, seq_num_by_step_num, seq_num_is_non_empty_loop_start, end_step_num
        )
        expected_constant_current_step = {
            "StepType": "Dischrge",
            "StepMode": "Current ",
            "StepValue": "100.00E-6",
            "Limits": "",
            "Ends": {
                "EndEntry": [
                    {
                        "EndType": "Voltage ",
                        "SpecialType": " ",
                        "Oper": ">= ",
                        "Step": "005",
                        "Value": "4.4",
                    }
                ],
            },
            "Reports": {"ReportEntry": [{"ReportType": "Current ", "Value": "2.2",}],},
            "Range": "4",
            "Option1": "N",
            "Option2": "N",
            "Option3": "N",
            "StepNote": "",
        }

        self._test_step_helper(expected_constant_current_step, constant_current_step)

    def test_biologic_mb_text_to_maccor_xml(self):
        xml = BiologicMbToMaccorProcedure.biologic_mb_text_to_maccor_xml(SAMPLE_MB_TEXT)
        xml_lines = xml.splitlines()
        expected_xml_lines = EXPECTED_XML.splitlines()
        for line_num in range(len(expected_xml_lines)):
            assert line_num < len(xml_lines)
            self.assertEqual(expected_xml_lines[line_num], xml_lines[line_num])

    def test_convert(self):
        # TODO
        #
        # Assert equivalence to a verified file
        source = os.path.join(TEST_FILE_DIR, SAMPLE_MB_FILE_NAME)
        target = os.path.join(TEST_FILE_DIR, CONVERTED_OUTPUT_FILE_NAME)
        BiologicMbToMaccorProcedure.convert(source, target)
        os.remove(target)
