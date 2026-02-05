# Introduction

![icon](static/tri_small.png)

![Testing - main](https://github.com/TRI-AMDD/beep/workflows/Testing%20-%20main/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
[![PyPI version](https://badge.fury.io/py/beep.svg)](https://badge.fury.io/py/beep)

BEEP is a set of tools designed to support Battery Evaluation and Early Prediction of cycle life corresponding to the research of the [d3batt program](https://d3batt.mit.edu/) and the [Toyota Research Institute](https://www.tri.global/our-work/energy-and-materials/).


BEEP enables parsing and handling of electrochemical battery cycling data
via data objects reflecting cycling run data, experimental protocols,
featurization, and modeling of cycle life with machine learning. Currently BEEP supports:

- [Arbin Instruments](https://www.arbin.com/) cyclers
- [Novonix Group](https://novonixgroup.com/) cyclers
- [MACCOR](http://www.maccor.com/) cyclers
- [BioLogic](https://www.biologic.net/product_category/battery-cyclers/) cyclers
- [Neware](https://newarebattery.com/) cyclers
- [Battery Archive](https://www.batteryarchive.org/) data

BEEP provides a standardized interface for working with cycler data ranging in scale
from a single file on a local laptop to running thousands of cycler files with massive
throughput on large computing systems.


We are currently looking for experienced python developers to help us improve this package and implement new features.
Please contact any of the maintainers for more information.


## Installation

### Basic Installation

Install from PyPI:

```bash
pip install beep
```

### Development Installation

For development, clone the repo and install with dev dependencies:

```bash
git clone https://github.com/TRI-AMDD/beep.git
cd beep
pip install -e ".[dev]"
```

This installs BEEP in editable mode along with:
- Testing tools (pytest, pytest-cov)
- Linting tools (ruff, mypy)
- Pre-commit hooks

### Setting Up Pre-commit Hooks (Optional)

To automatically run linting on each commit:

```bash
pre-commit install
```


## Requirements

- **Python**: 3.10, 3.11, or 3.12
- **Operating Systems**: Linux, macOS, Windows

Core dependencies are installed automatically:
- numpy, scipy, pandas, scikit-learn
- matplotlib, h5py
- boto3 (for S3 integration)
- click (for CLI)


## Testing

Run the test suite:

```bash
pytest beep
```

Run with coverage:

```bash
pytest beep --cov=beep --cov-report=html
```


## Quick Start

```python
from beep.structure.maccor import MaccorDatapath

# Load a cycler file
datapath = MaccorDatapath.from_file("path/to/your/file.txt")

# Structure the data
datapath.structure()

# Access structured data
print(datapath.structured_summary)
print(datapath.structured_data)

# Save for later use
datapath.to_json_file("structured_data.json.gz")
```


## How to cite

If you use BEEP, please cite this article:

> P. Herring, C. Balaji Gopal, M. Aykol, J.H. Montoya, A. Anapolsky, P.M. Attia, W. Gent, J.S. Hummelshøj, L. Hung, H.-K. Kwon, P. Moore, D. Schweigert, K.A. Severson, S. Suram, Z. Yang, R.D. Braatz, B.D. Storey, SoftwareX 11 (2020) 100506.
[https://doi.org/10.1016/j.softx.2020.100506](https://doi.org/10.1016/j.softx.2020.100506)
