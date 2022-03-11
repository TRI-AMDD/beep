# Contribution guidelines

Here are some simple contribution guidelines which you may find useful if you are considering adding features, bug fixes, or other updates to `beep`.

Working within these guidelines will help your pull request be added in a timely fashion!

Note this guide is meant primarily for battery scientists with limited experience in software development, particularly in python. 

**In general, our goal here is to implement ideas which can be complex, convoluted, or confusing in the simplest, most concise, and most reusable manner possible.**


## Code standards

### General

- **Don't write boilerplate code**: If repeated or similar lines of code can be encapsulated in a function, define a function and reuse the function.
- **Good code should be easily readable on a line by line basis**: Generally, we try to keep the total number of lines of code to a minimum. However, each line of code should be readable and as short as possible. If you are stuck choosing between fewer (yet more complex) lines of code and more (yet simpler), usually you should opt for more (yet simpler) code.
- **Don't use highly complex one-liners**: Although python allows for list and generator comprehensions in a single line, nesting these comprehensions or making them overly complex makes code very difficult to read. Often, it is easier to read an explicit `for` loop than to decompose a complex one-line comprehension or boolean condition.

Example 1:

```python
# Bad - incomprehensible at a glance
x = [t for t in {yi: yj for yi, yj in y_dict.items() if yi in yset}.values() if (t != 42 and t is not None)]

# Better - can at leas tunderstand each line at a glance
y = {}
for yi, yj in y_dict.items():
    if yi in yset:
        y[yi] = yj
x = []
for t in y.values()
    if t != 42 and not t:
        x.append(t)
```


Example 2:
```python
# Bad - impossible to read
if (t != 42 and t is not None) or (array[0] != 42 and int(abs(max(array)))) not in forbidden_arrays):
    do_something()
    
# Better - can at least understand each line at a glance
t_valid = t not in (42, None)
array_valid = array[0] != 42
array_forbidden = int(abs(max(array))) in forbidden_arrays
if t_valid or (array_valid and not array_forbidden):
    do_something()
```

- **Limit the number of arguments to functions/methods to ~10**: Usually methods or functions with more than 10 arguments are hard to read.
- **Use informative variable names**: Use properly formatted, minimal, and informative names for variables, functions, and module names. 

Example:

```python
# Bad - a few examples of uninformative or ambiguous variable names
output_value = V/R
OutputValue = V/R
CurrentValue = V/R


# Better - minimal and informative variable names
current = V/R
current_amps = voltage_volts/resistance_ohms
```


- **Use builtin libraries whenever possible**: [The python standard library](https://docs.python.org/3/library/) has many useful libraries. Usually, working with the standard library modules is a reasonably performant, well supported, and highly error tolerant solution. Using external libraries or writing your own alternatives to the standard library's functions are encouraged only when there are significant performance, usability, or code clarify advantages.
- **Use informative exceptions**: It is much easier to debug code with thoughtfully constructed exceptions (errors) than to reverse-engineer. For example, when an input is outside the expected range, use a `raise ValueError("Explanation goes here")`. 

### Formatting

- Adhering to any formatting standard (numpy-style, Google style, etc.) ensures your code can be read more easily.
- We encourage using the [Google python standard style guide](https://google.github.io/styleguide/pyguide.html)
- The [PEP8 style guidelines](https://peps.python.org/pep-0008/) can also be quite helpful for cleaning up your code.
- **Programmatic tools**: We encourage you to check your code with [`flake8`](https://flake8.pycqa.org/en/latest/) and [`pycodestyle`](https://pycodestyle.pycqa.org/en/latest/); these are the tools we use for automatically linting new pull requests.









## Documenting your code


Well-documented code ensures others can use and improve upon your code. 

### Inline comments

In general, your code should explain itself; it should not need clarification from additional comments. However, it is
occasionally necessary to add inline comments for explaining or citing particular methods, especially if those methods are
esoteric or not explained elsewhere. Here is an example of a good block comment:

```
# Regression according to Mathiesen's method; constants taken from 
# this publication: https://doi.org/10.101/12345
my_variable2 = (input1 * input2**2)/some_constant
final_answer = constant3 * my_variable2

```

### Docstrings and module comments

**It is imperative that each function, method, class, and module you write are comprehensively documented.** See [Section 3.8 of the Google Python style guide](https://google.github.io/styleguide/pyguide.html) for some examples of how to do this. 



## Writing unittests

Unittests are a way to check that your code works as intended. **Code with new functionality must have tests!** Testing your code means writing "test" methods which call a desired function/method with some known ground truth inputs and output. If the *real* output of your function/method matches the expected ground truth output, the test passes. 

In general, you should write unittests for each new functionality your code performs. Writing unittests at the same time you add a new piece of code (function, method, class) is the easiest way to do this.


The fundamental unit of unittesting is a `TestCase` class. A `TestCase` class holds a set of related tests. `TestCase`s go in modules specific for testing - for example, `beep.structure.test.test_validate` is a testing module. 

For more information on the syntax for checking the correctness of statements (e.g., `self.assertTrue`), see the [official python unittesting documentation.](https://docs.python.org/3/library/unittest.html)


### Step 1: Find the correct module for adding your tests

If your code is in an existing module (e.g., `beep.features.intracell_analysis`), your tests will go in that module's test module (e.g., `beep.features.tests.test_intracell_analysis`)

If your code is in a new module (e.g., `beep.structure.my_new_module`), your tests will go either:

- in a new module in that test directory (`beep.structure.tests.test_my_new_module`)
- in an existing module which implements tests for code similar to yours (e.g., if you are adding a new cycler datapath, `beep.structure.test_cyclerpaths`)

**If you are not sure where your test code should go, ask a developer in your pull request!**

### Step 2: Create one or more `TestCase`s 

A unittest `TestCase` is a set of methods which will run to test your new code. 

- If your contribution is a small bug fix, you will add testing code inside an existing `TestCase` class.
- If your contribution adds new methods to an existing class or new functions to an existing module, your tests will go inside an existing `TestCase` class.
- If you are adding a new class, your tests should go in a new `TestCase` class.
- If you are adding multiple new classes or a new module, your tests shoudl go in multiple `TestCase` classes. For example, if you are adding `Class1` and `Class2` as new classes, you should probably have `TestClass1` and `TestClass2` as `TestCases`.


### Step 3: Create one test method for each method or function in your `TestCase`s

Inside your `TestCase` class, implement some basic - yet realistic - test cases to ensure your code is working as intended. This is where you will use python's unittesting library's `self.assert*` methods to check the outputs of code for correctness.

If you are adding a class, there should be one testing method for each method of your new class.

If you are adding one or more functions, there should be one testing method for each function added.


Make sure your test cases work for:

- Minimal basic inputs with known outputs; ensure these tests are simple yet realistic.
- Edge cases which likely will be encountered (e.g., a numerical input is maximized, a numerical input is minimized, etc.)
- Erroneous input throws the expected exceptions using `self.assertRaises`



### Unittesting template

Here is a template/example of how to write unittests for a new class. The easiest way to get started is to copy+paste this code
and replace the code with our own tests.

```python

import unittest

from beep.my_new_module import MyNewClass


class TestMyNewClass(unittest.TestCase)
    def test_my_new_class(self):
        # testing the __init__ behavior of your class, for example
        inputs = ["A", 1, 15.2]
        mnc = MyNewClass(*inputs)
        
        self.assertTrue(mnc.some_attr)
        self.assertFalse(mnc.some_attr2)
    
    def test_compute(self):
        # testing a particular method "compute" of your "MyNewClass" 
        # class against a bunch of inputs

        mnc = MyNewClass("B", 2, 21.3)
        arg1 = SomeObject()
        x = range(1, 5)
        
        for i in x:
            self.assertEqual(mnc.compute(arg1, i), 10)
            self.assertAlmostEqual(mnc.compute(arg1, i, as_float=True), 9.999999)
            
        # Make sure compute fails in the way we expect
        with self.assertRaises(TypeError):
            mnc.compute(arg1, "bad_input")
```


### Step 4: Run your tests!

While all tests are checked by the Github continuous integration, you should run your tests locally. 

First, run your tests by themselves. Make sure you have the requirements from `requirements-test.txt` installed. You can then run your new test cases by  adding the following code at the bottom of the test file and running it.

```python
if __name__ == "__main__":
    # replace TestMyNewClass with your TestCase name!
    unittest.main(TestMyNewClass())
```

If your test passed, congrats!

You might also want to make sure your new code did not break any other tests. You can do this from the command line  in the base `beep` directory (the same directory as `setup.py`):

```shell
$: pytest beep --color=yes
```



### Some tips for writing tests

Find more info for each of these tips on [the python unittesting docs.](https://docs.python.org/3/library/unittest.html)

- You can define a special `setUp` method for performing the same setup actions (e.g., clearing or resetting class attributes, creating a common input file) for all of your test methods. This can cut down on your boilerplate code.
- You can define a special `setUpClass` class method which will run once before *any* of the test methods run. 
- You can define a special `tearDown` method for performing the same post-test actions after each test. This is useful for cleaning up leftover files. This is similar to `setUp`.
- You can define a special `tearDownClass` class method which will run once at the end of the `TestCase`. 


