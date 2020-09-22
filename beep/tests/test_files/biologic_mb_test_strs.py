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
