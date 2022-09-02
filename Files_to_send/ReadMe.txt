New Arbin files

Mits8.0
Software Version:     202110
Firmware Version:     TY2021.10.22.1



Here are the Arbin files that did not compile with the new code. They have different headings and lacked an epoch time.


---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
   3620             try:
-> 3621                 return self._engine.get_loc(casted_key)
   3622             except KeyError as err:

~/opt/anaconda3/lib/python3.8/site-packages/pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()

~/opt/anaconda3/lib/python3.8/site-packages/pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'date_time'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
<ipython-input-2-df00088479f9> in <module>
     13 #datapath = ArbinDatapath.from_file("/Users/guilhermemissaka/Downloads/2017-05-12_6C-50per_3_6C_CH36.csv")
     14 #datapath = ArbinDatapath.from_file("/Users/guilhermemissaka/Desktop/Research/arbinCode/Johnson.csv")
---> 15 datapath = ArbinDatapath.from_file("/Users/guilhermemissaka/Desktop/Files_to_send/LFP95_thick_08162022_wab35_Channel_2_Wb_1.csv")
     16 
...
-> 3623                 raise KeyError(key) from err
   3624             except TypeError:
   3625                 # If we have a listlike key, _check_indexing_error will raise

KeyError: 'date_time'

————————————————————————————————————————————————————————————————————————————



There is also a python file we wrote that converts the epoch time and headings into the expected results. However, it does still has a problem with the interpolate step in base.py (in structure line 1419) where it does not work with “slinear”  interpolation, but does with “linear.” 



————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————


---> 25 features = datapath.structure()
     26 #print(datapath.structuring_parameters)
     27 #print(datapath.is_structured)

~/Desktop/Research/arbinCode/beep/structure/base.py in wrapper(*args, **kwargs)
    127                     )
    128                 else:
--> 129                     return func(*args, **kwargs)
    130 
    131             return wrapper

~/Desktop/Research/arbinCode/beep/structure/base.py in structure(self, v_range, resolution, diagnostic_resolution, nominal_capacity, full_fast_charge, diagnostic_available, charge_axis, discharge_axis)
    495             )
    496 
--> 497         self.structured_data = self.interpolate_cycles(
    498             v_range=v_range,
    499             resolution=resolution,

~/Desktop/Research/arbinCode/beep/structure/base.py in interpolate_cycles(self, v_range, resolution, diagnostic_available, charge_axis, discharge_axis)
    714 
...
--> 536             raise ValueError("x and y arrays must have at "
    537                              "least %d entries" % minval)
    538 

ValueError: x and y arrays must have at least 2 entries