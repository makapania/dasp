---
paths: **/*.py
---
# Python Code Style

## Formatting
- Use Black with line-length 100 (configured in pyproject.toml)
- Sort imports: stdlib, third-party, local

## Type Hints
- Add type hints to all new functions
- Use `from __future__ import annotations` for forward references
- Prefer built-in generics (`list[str]` not `List[str]`)

## Naming
- snake_case for functions and variables
- PascalCase for classes
- SCREAMING_SNAKE_CASE for constants
- Prefix private methods with `_`

## Documentation
- Docstrings for public functions/classes (Google style)
- Only add comments for non-obvious logic
- Keep docstrings concise

## Scientific Computing Conventions
- Use numpy for array operations
- Use pandas for tabular data
- Prefer vectorized operations over loops
- Use scipy.signal for spectral preprocessing

## Error Handling
- Use specific exception types
- Don't catch bare `except:`
- Log errors with context using logging module
