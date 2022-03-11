# Contribution guidelines

Here are some simple contribution guidelines which you may find useful if you are considering adding features, bug fixes, or other updates to `beep`.

Working within these guidelines will help your pull request be added in a timely fashion!

Note this guide is meant primarily for battery scientists with limited experience in software development, particularly in python. 

**In general, our goal here is to write code** (which can complex, convoluted, or confusing  things) **in the absolute simplest, most concise, and reusable manner possible.**


## Code standards

### General

- **Don't write boilerplate code**: If repeated or similar lines of code can be encapsulated in a function, define a function and reuse the function.
- **Good code should be easily readable on a line by line basis**: Generally, we try to keep the total number of lines of code to a minimum. However, each line of code should be readable and as short as possible. If you are stuck choosing between fewer (yet more complex) lines of code and more (yet simpler), usually you should opt for more (yet simpler) code.
- **Don't use highly complex one-liners**: Although python allows for list and generator comprehensions in a single line, nesting these comprehensions or making them overly complex makes code very difficult to read. Often, it is easier to read an explicit `for` loop than to decompose a complex one-line comprehension or boolean condition.

Example 1:

```python
# Bad - incomprehensible at a glance
x = [t for t in {yi: yj for yi, yj in y_dict.items() if yi in yset}.values() if (t != 42 and t is not None)]

# Better - can at leas understand each line at a glance
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
final_answer = constant3 * 

```

### Docstrings and module comments

It is imperative that each function, method, class, and module you write are comprehensively documented. See [Section 3.8 of the Google Python style guide](https://google.github.io/styleguide/pyguide.html) for some examples of how to do this. 



## Writing unittests

In general, you should write unittests for each new functionality your code performs. 