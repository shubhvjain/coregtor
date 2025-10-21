# Contributing to CoRegTor


### Naming Conventions

- **Functions**: Use `snake_case` with descriptive verb prefixes
- **Variables**: Use `snake_case` (eg  `expression_data`, `target_gene`, `feature_importance`)
- **Classes**: Use `PascalCase`
- **Constants**: Use `UPPER_CASE` ( eg `DEFAULT_N_ESTIMATORS = 100`)

### Function Design

- Functions should do one thing well
- Use type hints for all parameters and return values
- Include  docstrings ([Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings))

```python
def function_name(param1, param2):
    """Short description.

    Longer description if needed. Include context or purpose.

    Args:
        param1 (type): Description of param1.
        param2 (type): Description of param2.

    Returns:
        type: Description of return value.

    Raises:
        ErrorType: Condition when raised.

    Examples:
        >>> function_name(2, 3)
        expected_output
        >>> function_name(5, 1)
        another_output
    """

```
All public functions must include:
- One-line summary
- Detailed description
- Args section with types
- Returns section with type
- Examples section
