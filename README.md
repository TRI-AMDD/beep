# Battery Evaluation and Early Prediction (BEEP)

<h4 align="center">

![Testing - main](https://github.com/TRI-AMDD/beep/workflows/Testing%20-%20main/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)
[![PyPI version](https://badge.fury.io/py/beep.svg)](https://badge.fury.io/py/beep)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/TRI-AMDD/beep?label=Repo+Size)](https://github.com/TRI-AMDD/beep/graphs/contributors)

</h4>

BEEP is a set of tools designed to support Battery Evaluation and Early Prediction of cycle life corresponding to the research of the [d3batt program](https://d3batt.mit.edu/) and the [Toyota Research Institute](http://www.tri.global/accelerated-materials-design-and-discovery/).

* **Documentation:** https://tri-amdd.github.io/beep
* **Source code:** https://github.com/TRI-AMDD/beep
* **PyPI release:** https://pypi.org/project/beep/

## Installation

```bash
pip install beep
```

For development:

```bash
git clone https://github.com/TRI-AMDD/beep.git
cd beep
pip install -e ".[dev]"
```

## Supported Cyclers

- [Arbin Instruments](https://www.arbin.com/)
- [Novonix Group](https://novonixgroup.com/)
- [MACCOR](http://www.maccor.com/)
- [BioLogic](https://www.biologic.net/product_category/battery-cyclers/)
- [Neware](https://newarebattery.com/)
- [Battery Archive](https://www.batteryarchive.org/) data

## Quick Start

```python
from beep.structure.maccor import MaccorDatapath

# Load and structure a cycler file
datapath = MaccorDatapath.from_file("path/to/file.txt")
datapath.structure()

# Access structured data
print(datapath.structured_summary)
```

## How to cite

If you use BEEP, please cite this article:

> P. Herring, C. Balaji Gopal, M. Aykol, J.H. Montoya, A. Anapolsky, P.M. Attia, W. Gent, J.S. Hummelshøj, L. Hung, H.-K. Kwon, P. Moore, D. Schweigert, K.A. Severson, S. Suram, Z. Yang, R.D. Braatz, B.D. Storey, SoftwareX 11 (2020) 100506.
[https://doi.org/10.1016/j.softx.2020.100506](https://doi.org/10.1016/j.softx.2020.100506)
