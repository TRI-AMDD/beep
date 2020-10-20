# Introduction

![icon](static/tri_small.png)

![Testing - main](https://github.com/TRI-AMDD/beep/workflows/Testing%20-%20main/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/TRI-AMDD/beep/badge.svg?branch=master)](https://coveralls.io/github/TRI-AMDD/beep?branch=master)

BEEP is a set of tools designed to support Battery Evaluation and Early Prediction of cycle life corresponding to the research of the [d3batt program](https://d3batt.mit.edu/) and the [Toyota Research Institute](http://www.tri.global/accelerated-materials-design-and-discovery/).


BEEP enables parsing and handing of electrochemical battery cycling data
via data objects reflecting cycling run data, experimental protocols,
featurization, and modeling of cycle life.  Currently beep supports 
arbin, maccor and biologic cyclers.

We are currently looking for experienced python developers to help us improve this package and implement new features.
Please contact any of the maintainers for more information.


## Installation
Use `pip install beep` to install.

If you want to develop BEEP, clone the repo via git and use 
pip (or `python setup.py develop`)  for an editable install:

```bash
git clone git@github.com:ToyotaResearchInstitute/BEEP.git
cd BEEP
pip install -e .[tests]
```
## Environment
To configure the use of AWS resources its necessary to set the environment variable `BEEP_ENV`. For most users `'dev'`
is the appropriate choice since it assumes that no AWS resources are available. 
```.env
export BEEP_ENV='dev'
```
For processing file locally its necessary to configure the folder structure 
```.env
export BEEP_PROCESSING_DIR='/path/to/beep/data/'
```

## Testing
You can use pytest for running unittests. In order to run tests the environment variable
needs to be set (i.e. `export BEEP_ENV='dev'`)

```bash
pytest beep
```


## How to cite
If you use BEEP, please cite this article:

> P. Herring, C. Balaji Gopal, M. Aykol, J.H. Montoya, A. Anapolsky, P.M. Attia, W. Gent, J.S. Hummelshøj, L. Hung, H.-K. Kwon, P. Moore, D. Schweigert, K.A. Severson, S. Suram, Z. Yang, R.D. Braatz, B.D. Storey, SoftwareX 11 (2020) 100506.
[https://doi.org/10.1016/j.softx.2020.100506](https://doi.org/10.1016/j.softx.2020.100506)

