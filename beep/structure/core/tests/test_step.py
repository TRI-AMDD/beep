"""
Copyright [2020] [Toyota Research Institute]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import copy
import unittest

import pandas as pd

from beep.structure.core.step import Step, MultiStep
from beep.structure.core.tests import DIR_TESTS_STRUCTURE_CORE


class TestStep(unittest.TestCase):

    def setUpClass(cls):
        cls.df_chg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "step_charge.csv"))
        cls.df_dchg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "step_discharge.csv"))
        cls.df_rest = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "step_rest.csv"))
        cls.step_class = Step

    def test_instantiation(self):
        print(self.df_chg)

    def test_uniqueness(self):
        pass

    def test_config(self):
        pass

    def test_as_dict(self):
        pass

    def test_from_dict(self):
        pass


class TestMultiStep(TestStep):

    def setUpClass(cls):
        cls.df_chg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "multistep_charge.csv"))
        cls.df_dchg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "multistep_discharge.csv"))
        cls.df_rest = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "multistep_rest.csv"))