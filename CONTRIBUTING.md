# Contributing to Lyrics Generator

Thank you for your interest in contributing to the Lyrics Generator project! We welcome contributions from everyone. This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

Please be respectful and professional in all interactions. We are committed to providing a welcoming and inclusive environment for all contributors.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lyrics_generator_generative_ai.git
   cd lyrics_generator
   ```
3. **Add** upstream remote to keep sync with main repo:
   ```bash
   git remote add upstream https://github.com/Mayankvlog/lyrics_generator_generative_ai.git
   ```
4. **Create** a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites
- Python 3.11+
- Git
- Virtual environment tool (venv)

### Step-by-Step Setup

1. **Create virtual environment:**
   ```bash
   python -m venv myenv
   
   # Windows
   myenv\Scripts\Activate.ps1
   
   # macOS/Linux
   source myenv/bin/activate
   ```

2. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8 mypy
   ```

3. **Set up pre-commit hooks (optional but recommended):**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Create .env file:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Making Changes

### Branch Naming Convention

Use descriptive branch names:
- `feature/add-xyz` - New features
- `fix/bug-description` - Bug fixes
- `docs/update-readme` - Documentation updates
- `refactor/code-section` - Code refactoring
- `test/add-tests` - Adding tests

### Code Style

We follow PEP 8 with these additional guidelines:

**Python Formatting:**
```bash
# Format code with Black
black main.py lyrics_generator.ipynb

# Check style with Flake8
flake8 *.py

# Type checking with MyPy
mypy main.py
```

**Code Example:**
```python
def generate_lyrics(model, tokenizer, seed_text, max_len, 
                   num_words: int = 100, 
                   temperature: float = 0.8) -> str:
    """
    Generate new lyrics using the trained model.
    
    Args:
        model: Trained Keras model
        tokenizer: Fitted tokenizer
        seed_text: Starting text for generation
        max_len: Maximum sequence length
        num_words: Number of words to generate (default: 100)
        temperature: Sampling temperature (default: 0.8)
        
    Returns:
        Generated lyrics as string
        
    Raises:
        ValueError: If inputs are invalid
        
    Example:
        >>> lyrics = generate_lyrics(model, tokenizer, "love", 100)
    """
    # Implementation here
    pass
```

### Docstring Format

Use Google-style docstrings:

```python
def retrieve_lyric(prompt: str, vectorizer, matrix, 
                  dataframe) -> str:
    """
    Retrieve the most relevant lyric from the dataset.
    
    Uses TF-IDF and cosine similarity to find semantically
    similar lyrics based on the input prompt.
    
    Args:
        prompt: User input query
        vectorizer: Fitted TF-IDF vectorizer
        matrix: TF-IDF matrix of dataset
        dataframe: DataFrame containing lyrics
        
    Returns:
        Most similar lyric from the dataset
        
    Raises:
        ValueError: If dataframe is empty
        TypeError: If vectorizer is not fitted
    """
    pass
```

## Submitting Changes

### Step 1: Keep your branch updated

```bash
git fetch upstream
git rebase upstream/main
```

### Step 2: Make your changes

- Make logical, atomic commits
- Test your changes thoroughly
- Update documentation as needed

### Step 3: Push to your fork

```bash
git push origin feature/your-feature-name
```

### Step 4: Create a Pull Request

1. Go to GitHub and create a Pull Request
2. Use a clear, descriptive title
3. Provide a detailed description of changes
4. Reference any related issues (e.g., "Fixes #123")
5. Ensure CI/CD checks pass

## Coding Standards

### Naming Conventions

```python
# Variables and functions: snake_case
max_sequence_length = 100
def generate_lyrics(): pass

# Classes: PascalCase
class LyricsGenerator: pass

# Constants: UPPER_SNAKE_CASE
DEFAULT_TEMPERATURE = 0.8

# Private members: _leading_underscore
def _internal_helper(): pass
```

### Comments and Documentation

```python
# Bad: Obvious comments
x = 5  # Set x to 5

# Good: Explain the "why"
max_sequence_length = 100  # Model was trained with 100-token sequences

# Documentation comments for complex logic
# This padding ensures all sequences match the training sequence length
# which is required by the Keras model for batch processing
token_list = pad_sequences([token_list], maxlen=effective_max_len)
```

### Imports

```python
# Standard library
import os
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Local imports
from config import DEFAULT_TEMPERATURE
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_generation.py

# Run with verbose output
pytest -v
```

### Writing Tests

```python
# tests/test_generation.py
import pytest
from main import generate_lyrics, retrieve_lyric

def test_generate_lyrics_returns_string():
    """Test that generate_lyrics returns a string."""
    # Setup
    mock_model = ...
    mock_tokenizer = ...
    
    # Execute
    result = generate_lyrics(mock_model, mock_tokenizer, "love", 100)
    
    # Assert
    assert isinstance(result, str)
    assert len(result) > 0

def test_retrieve_lyric_empty_dataframe():
    """Test that retrieve_lyric raises error on empty dataframe."""
    with pytest.raises(ValueError):
        retrieve_lyric("test", vectorizer, matrix, pd.DataFrame())
```

## Commit Messages

Use clear, descriptive commit messages following the conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring without feature changes
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Build process, dependencies, etc.

### Examples

```bash
# Feature
git commit -m "feat(generation): add temperature control for output diversity"

# Bug fix
git commit -m "fix(data-loading): handle missing Lyric column gracefully"

# Documentation
git commit -m "docs(readme): add Docker deployment instructions"

# Refactor
git commit -m "refactor(model-loading): extract artifacts loading to separate function"
```

## Pull Request Process

1. **Before submitting:**
   - ✅ All tests pass
   - ✅ Code is formatted (Black)
   - ✅ No linting errors (Flake8)
   - ✅ Type checking passes (MyPy)
   - ✅ Documentation is updated
   - ✅ Commit history is clean

2. **PR Description template:**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Refactoring
   
   ## Related Issues
   Closes #(issue number)
   
   ## Testing
   How to test these changes
   
   ## Screenshots (if applicable)
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] All tests passing
   ```

3. **After submitting:**
   - Address review feedback
   - Push updates to the same branch
   - Respond to all comments
   - Request re-review when ready

## Reporting Bugs

When reporting bugs, include:
- Python version and OS
- Exact steps to reproduce
- Expected vs actual behavior
- Error messages and tracebacks
- Screenshots (if applicable)
- Your environment details

## Suggesting Features

When suggesting features:
- Explain the use case
- Provide examples of how it would work
- Explain why it would be beneficial
- Consider potential implementation challenges

## Questions?

Feel free to:
- Open a GitHub Discussion
- Create an issue with your question
- Contact: mayankkr0311@gmail.com

## Recognition

Contributors will be recognized in the project README under the "Contributors" section.

Thank you for contributing! 🎉
