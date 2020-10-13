BT-LAB SETTING FILE

Number of linked techniques : 1

Filename : C:\Users\sample.mps

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
ctrl3_val_vs        1.700               1.700               1t.700               1.700               
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
