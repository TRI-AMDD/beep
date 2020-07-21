# Contributing to BEEP

We are striving for community-driven adoption and development of BEEP. If you would like to 
contribute (thank you!), please have a look at the [guidelines below](#workflow).

If you're already familiar with our workflow, maybe have a quick look at the [pre-commit checks](#pre-commit-checks) directly below.

## Pre-commit checks

Before you commit any code, please perform the following checks:

- [All tests pass](#testing): `$ nosetests beep`
- [No style issues](#coding-style-guidelines): `$ flake8`

## Workflow

We use [GitHub](https://en.wikipedia.org/wiki/GitHub) standard [Fork and Pull Request workflow](https://gist.github.com/Chaser324/ce0505fbed06b947d962) to coordinate our work. 
When making any kind of update, we try to follow the procedure below.

### A. Before you begin

1. Create an [issue](https://guides.github.com/features/issues/) where new proposals can be discussed before any coding is done.
2. Create a personal [fork](https://help.github.com/articles/fork-a-repo/) the master repo.
3. Download the source code onto your local system, by [cloning](https://help.github.com/articles/cloning-a-repository/) the repository (or your fork of the repository).
4. Create a [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/) of this repo on your own fork where all changes will be made
5. [Install](#installation) BEEP with the developer options.
6. Test if your installation worked. `nosetests beep`.

You now have everything you need to start making changes!

### B. Writing your code

7. BEEP is developed in [Python](https://en.wikipedia.org/wiki/Python_(programming_language). Make sure to follow our [coding style guidelines](#coding-style-guidelines).
8. Commit your changes to your branch with [useful, descriptive commit messages](https://chris.beams.io/posts/git-commit/). While developing, you can keep using the GitHub issue you're working on as a place for discussion. [Refer to your commits](https://stackoverflow.com/questions/8910271/how-can-i-reference-a-commit-in-an-issue-comment-on-github) when discussing specific lines of code.
9. If you want to add a dependency on another library, or re-use code you found somewhere else, have a look at [these guidelines](#dependencies-and-reusing-code).

### C. Merging your changes with BEEP

9. [Test your code!](#testing)
11. If you added a major new feature, perhaps it should be showcased in an [example notebook](#example-notebooks).
12. When you feel your code is finished, or at least warrants serious discussion, run the [pre-commit checks](#pre-commit-checks) and then create a [pull request](https://help.github.com/articles/about-pull-requests/).
13. Once a PR has been created, it will be reviewed by the maintainers. Changes might be suggested which you can make by simply adding new commits to the branch. When everything's finished, someone with the right GitHub permissions will merge your changes into BEEP repository.


## Installation

To install BEEP with all developer options, type:

```bash
pip install -e .[dev,docs]
```

This will

1. Install all the dependencies for BEEP, including the ones for documentation (docs) and development (dev).
2. Tell Python to use your local beep files when you use `import beep` anywhere on your system.

## Coding style guidelines

BEEP follows the [PEP8 recommendations](https://www.python.org/dev/peps/pep-0008/) for coding style. These are very common guidelines, and community tools have been developed to check how well projects implement them.

### Flake8

We use [flake8](http://flake8.pycqa.org/en/latest/) to check our PEP8 adherence. To try this on your system, navigate to the BEEP directory in a console and type

```bash
flake8
```
The configuration file
```
.flake8
```
allows us to ignore some errors. If you think this should be added or removed, please submit an [issue](#issues)

When you commit your changes they will be checked against flake8 automatically (see [infrastructure](#infrastructure)).


### Black

We use [black](https://black.readthedocs.io/en/stable/) to automatically configure our code to adhere to PEP8. Black can be used in two ways:

1. Command line: navigate to the BEEP directory in a console and type

```bash
black {source_file_or_directory}
```

2. Editor: black can be [configured](https://black.readthedocs.io/en/stable/editor_integration.html) to automatically reformat a python script each time the script is saved in an editor.

If you want to use black in your editor, you may need to change the max line length in your editor settings.

Even when code has been formatted by black, you should still make sure that it adheres to the PEP8 standard set by [Flake8](#flake8).

### Naming

In general, we aim for descriptive class, method, and argument names.
2. Avoid abbreviations when possible without making names overly long, so `mean` is better than `mu`, but a class name like `MyClass` is fine.
3. Class names are CamelCase, and start with an upper case letter, for example `FastChargeFeatures` or `ProcessedCyclerRun`. 
4. Method and variable names are lower case, and use underscores for word separation, for example `x` or `calculate_cycle_life`.

### Docstrings
Every method and every class should have a [docstring](https://www.python.org/dev/peps/pep-0257/) that
 describes in plain terms what it does, and what the expected input and output is. Likewise, while performing
 any complex operations in a piece of code, or putting in decision logic, please use in-line comments.

## Dependencies and reusing code

While it's a bad idea for developers to "reinvent the wheel", it's important for users to get a _reasonably sized download and an easy install_. In addition, external libraries can sometimes cease to be supported, and when they contain bugs it might take a while before fixes become available as automatic downloads to BEEP users.
For these reasons, all dependencies in BEEP should be thought about carefully, and discussed on GitHub.

Direct inclusion of code from other packages is possible, as long as their license permits it and is compatible with ours, but again should be considered carefully and discussed in the group. Snippets from blogs and [stackoverflow](https://stackoverflow.com/) can often be included without attribution, but if they solve a particularly nasty problem (or are very hard to read) it's often a good idea to attribute (and document) them, by making a comment with a link in the source code.


### Matplotlib

We use Matplotlib in BEEP, but with two caveats:

First, Matplotlib should only be used in plotting methods, and these should _never_ be called by other BEEP methods. So users who don't like Matplotlib will not be forced to use it in any way. Use in notebooks is OK and encouraged.

Second, Matplotlib should never be imported at the module level, but always inside methods. For example:

```
def plot_great_things(self, x, y, z):
    import matplotlib.pyplot as pl
    ...
```

This allows people to (1) use BEEP without ever importing Matplotlib and (2) configure Matplotlib's back-end in their scripts, which _must_ be done before e.g. `pyplot` is first imported.


## Testing

All code requires testing. We use the nosetests package for our tests. IF nose is not installed yet, run
```bash 
pip install nose
``` 
### Writing tests

Every new feature should have its own test. To create ones, have a look at the `test` directory and see if there's a test for a similar method. Copy-pasting this is a good way to start.

Next, add some simple (and speedy!) tests of your main features. If these run without exceptions that's a good start! Next, check the output of your methods using any of these [assert methods](https://docs.python.org/3.3/library/unittest.html#assert-methods).

### Running tests

The tests are divided into `unit` tests, whose aim is to check individual bits of code, and end-to-end tests, which check how parts of the program interact as a whole.

To run all tests,
```bash
nosetests beep
```
To run a specific test script
```bash
nosetests test_featurize.py
```
To run a specific test from a test class,
``` bash
nosetests test_structure.py:TestFeaturizer.test_insufficient_data_file
```
If you want to run several, but not all, the tests from a script, you can restrict which tests are run from a particular script by using the skipping decorator:
```python
@unittest.skip("")
def test_bit_of_code(self):
    ...
```
or by just commenting out all the tests you don't want to run

### Debugging tests

-  Set break points, either in your IDE or using the python debugging module. To use the latter, add the following line where you want to set the break point This will start the [Python interactive debugger](https://gist.github.com/mono0926/6326015).
```python
import ipdb; ipdb.set_trace()
```
- Warnings: If functions are raising warnings instead of errors, it can be hard to pinpoint where this is coming from. Here, you can use the `warnings` module to convert warnings to errors:
  ```python
  import warnings
  warnings.simplefilter("error")
  ```
  Then you can use a try-except block, as in a., but with, for example, `RuntimeWarning` instead of `ValueError`.

### Profiling


## Documentation

### Building the documentation


### Tutorial notebooks

Specific use-cases and capabilities of BEEP are showcased in Jupyter notebooks stored in the [examples directory](beep/tutorials). 
All example notebooks should be listed in [tutorials/README.md](https://github.com/TRI-AMDD/beep/tree/master/beep/tutorials). 


## Infrastructure

### Setuptools

Installation _and dependencies_ are handled via [setuptools](http://setuptools.readthedocs.io/)

Configuration files:

```
setup.py
```

Note that this file must be kept in sync with the version number in [beep/__init__.py](beep/__init__.py).

### Continuous integration

All committed code is tested using [Travis CI](https://travis-ci.org/), tests are published [here](https://travis-ci.com/github/TRI-AMDD/beep/).
Configuration file: ``` .travis.yaml```. For every commit, Travis runs all the unit tests, end to end tests, doc tests and flake8.
<!-- Unit tests and flake8 testing is done for every commit. A nightly cronjob also tests the notebooks. Notebooks listed in `.slow-books` are excluded from these tests. -->

Additionally, Appveyor runs integration tests for windows environment. Tests are published [here](https://ci.appveyor.com/project/TRI-AMDD/beep)


### Codecov

Code coverage (how much of the code is actually seen by the (linux) unit tests) is tested using [Codecov](https://docs.codecov.io/).

Configuration file: ``` .coveragerc```


## Acknowledgements

This CONTRIBUTING.md file was adapted from [Pints GitHub repo](https://github.com/pints-team/pints).
