# Contribution guidelines

Here are some simple contribution guidelines which you may find useful if you are considering adding features, bug fixes, or other updates to `beep`.

Working within these guidelines will help your pull request be added in a timely fashion!

Note this guide is meant primarily for battery scientists with limited experience in software development, particularly in python. 


## Code standards

#### General

- **Don't write boilerplate code**: If repeated or similar lines of code can be encapsulated in a function, define a function and reuse the function.
- **Good code should be easily readable on a line by line basis**: Generally, we try to keep the total number of lines of code to a minimum. However, each line of code should be readable and as short as possible. If you are stuck choosing between fewer (yet more complex) lines of code and more (yet simpler), usually you should opt for more (yet simpler) code.
- **Don't use highly complex one-liners**: Although python allows for list and generator comprehensions in a single line, nesting these comprehensions or making them overly complex makes code very difficult to read. Often, it is easier to read an explicit `for` loop than to decompose a complex one-line comprehension or boolean condition.

Example 1:

```python
# Bad
x = [t for t in {yi: yj for yi, yj in y_dict.items() if yi in yset}.values() if (t != 42 and t is not None)]

# Better
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
# Bad
if (t != 42 and t is not None) or (array[0] != 42 and int(abs(max(array)))) not in forbidden_arrays):
    do_something()
    
# Better
t_valid = t not in (42, None)
array_valid = array[0] != 42
array_forbidden = int(abs(max(array))) in forbidden_arrays
if t_valid or (array_valid and not array_forbidden):
    do_something()
```

- **Limit the number of arguments to functions/methods to ~10**: Usually methods or functions with more than 10 arguments are hard to read.
- 

#### Organization



#### Formatting









## Documenting your code


Well-documented code ensures others can use and improve upon your code. 

#### Inline comments

In general, your code should explain itself; it should not need clarification from additional comments. However, it is
occasionally necessary to add inline comments for explaining or citing particular methods, especially if those methods are
esoteric or not explained elsewhere. Here is an example of a good block comment:



```
# Regression according to Mathiesen's method; constants taken from 
# this publication: https://doi.org/10.101/12345
my_variable2 = (input1 * input2**2)/some_constant
final_answer = constant3 * 

```




## Writing unittests