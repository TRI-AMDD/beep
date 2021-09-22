# Introduction

![icon](static/tri_small.png)

![Testing - main](https://github.com/TRI-AMDD/beep/workflows/Testing%20-%20main/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/TRI-AMDD/beep/badge.svg?branch=master)](https://coveralls.io/github/TRI-AMDD/beep?branch=master)

BEEP is a set of tools designed to support Battery Evaluation and Early Prediction of cycle life corresponding to the research of the [d3batt program](https://d3batt.mit.edu/) and the [Toyota Research Institute](http://www.tri.global/accelerated-materials-design-and-discovery/).


BEEP enables parsing and handling of electrochemical battery cycling data
via data objects reflecting cycling run data, experimental protocols,
featurization, and modeling of cycle life with machine learning.  Currently BEEP supports:

- [Arbin Instruments](https://www.arbin.com/) cyclers
- [MACCOR](http://www.maccor.com/) cyclers
- [BioLogic](https://www.biologic.net/product_category/battery-cyclers/) cyclers
- [Battery Archive](https://www.batteryarchive.org/) data

With partial and forthcoming support for:

- Indigo cyclers
- [Neware](https://newarebattery.com/) cyclers


BEEP provides a standardized interface for working with cycler data ranging in scale
from a single file on a local laptop to running thousands of cycler files with massive
throughput on large computing systems.


We are currently looking for experienced python developers to help us improve this package and implement new features.
Please contact any of the maintainers for more information.


## Installation
To a base install, do:

```bash
pip install beep
```

If you want to develop BEEP and run tests, clone the repo via git and use 
pip (or `python setup.py develop`)  for an editable install:

```bash
git clone git@github.com:ToyotaResearchInstitute/BEEP.git
cd BEEP
pip install -e .[tests]
```


## Testing
Make sure you have installed the required testing packages (see installation).

```bash
pytest beep
```


## How to cite
If you use BEEP, please cite this article:

> P. Herring, C. Balaji Gopal, M. Aykol, J.H. Montoya, A. Anapolsky, P.M. Attia, W. Gent, J.S. Hummelshøj, L. Hung, H.-K. Kwon, P. Moore, D. Schweigert, K.A. Severson, S. Suram, Z. Yang, R.D. Braatz, B.D. Storey, SoftwareX 11 (2020) 100506.
[https://doi.org/10.1016/j.softx.2020.100506](https://doi.org/10.1016/j.softx.2020.100506)

