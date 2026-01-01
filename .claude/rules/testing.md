---
paths: tests/**/*.py,**/*_test.py,**/test_*.py
---
# Python Testing Rules

## Framework
- Use pytest for all tests
- Test files in `tests/` directory

## Structure
- Mirror src/ structure in tests/
- One test file per module
- Use descriptive test names: `test_function_does_specific_thing`

## Running Tests
```bash
pytest tests/ -v
pytest tests/test_specific.py -v  # Single file
pytest tests/ -k "keyword" -v     # Filter by name
```

## Fixtures
- Use pytest fixtures for setup/teardown
- Prefer factory fixtures over complex setup
- Scope fixtures appropriately (function, class, module, session)

## Assertions
- One assertion per test when possible
- Use pytest's assert rewriting
- Test edge cases and error conditions

## Mocking
- Use pytest-mock or unittest.mock
- Mock at boundaries (file I/O, external APIs)
- Don't over-mock - test real behavior when possible

## Data
- Use example data in `example/` directory for integration tests
- Create minimal synthetic data for unit tests
