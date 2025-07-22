# Contributing to Blackbird Open Source SDK

Thank you for your interest in contributing to the Blackbird Open Source SDK! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Report bugs and issues
- **Feature Requests**: Suggest new features
- **Code Contributions**: Submit code improvements
- **Documentation**: Improve documentation
- **Examples**: Add new examples
- **Testing**: Help with testing and quality assurance

## 🚀 Getting Started

### Prerequisites

- Python 3.8-3.11
- Git
- Basic understanding of Python and AI/ML concepts

### Development Setup

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/blackbird-open-source-sdk.git
   cd blackbird-open-source-sdk
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # Install platform-specific requirements
   pip install -r requirements_windows.txt  # Windows
   pip install -r requirements_mac.txt      # macOS
   pip install -r requirements.txt          # Linux
   ```

4. **Install Development Dependencies**
   ```bash
   pip install pytest flake8 black isort
   ```

5. **Verify Setup**
   ```bash
   python examples/test_sdk_minimal.py
   ```

## 📝 Development Guidelines

### Code Style

We follow PEP 8 with some modifications:

- **Line Length**: 88 characters (Black formatter)
- **Imports**: Use absolute imports for external packages, relative imports for internal modules
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Use type hints for all function parameters and return values

### Code Formatting

We use automated tools for code formatting:

```bash
# Format code
black open_source_sdk/
isort open_source_sdk/

# Check code style
flake8 open_source_sdk/
```

### Testing

Write tests for new features and ensure existing tests pass:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_specific_feature.py

# Run with coverage
pytest --cov=open_source_sdk tests/
```

### Documentation

- Update docstrings for new functions and classes
- Update README.md if adding new features
- Add examples for new functionality
- Update API documentation

## 🐛 Bug Reports

### Before Submitting a Bug Report

1. Check existing issues to avoid duplicates
2. Try to reproduce the issue with the latest version
3. Check the troubleshooting section in STARTUP_GUIDE.md

### Bug Report Template

```markdown
**Bug Description**
Brief description of the issue

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Environment**
- OS: [Windows/macOS/Linux]
- Python Version: [3.8/3.9/3.10/3.11]
- SDK Version: [if applicable]
- Hardware: [CPU/GPU specs]

**Additional Information**
- Error messages
- Screenshots
- Logs
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Brief description of the feature

**Use Case**
Why this feature would be useful

**Proposed Implementation**
How you think it could be implemented (optional)

**Alternatives Considered**
Other approaches you've considered (optional)
```

## 🔧 Code Contributions

### Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation

3. **Test Your Changes**
   ```bash
   # Run tests
   pytest tests/
   
   # Run examples
   python examples/test_sdk_minimal.py
   python examples/chat_response.py
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

We use conventional commit messages:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Maintenance tasks

### Pull Request Template

```markdown
**Description**
Brief description of changes

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Example addition
- [ ] Other (please describe)

**Testing**
- [ ] Tests pass
- [ ] Examples work
- [ ] Documentation updated

**Screenshots**
If applicable, add screenshots

**Additional Notes**
Any additional information
```

## 📚 Documentation Contributions

### Documentation Guidelines

- Use clear, concise language
- Include code examples
- Keep documentation up to date
- Use proper markdown formatting

### Documentation Structure

```
docs/
├── README.md              # Main documentation
├── STARTUP_GUIDE.md       # Installation and setup
├── examples/
│   ├── README.md          # Examples documentation
│   └── *.py              # Code examples
└── api/                   # API documentation
```

## 🧪 Testing Guidelines

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies

### Test Structure

```python
def test_feature_name():
    """Test description."""
    # Arrange
    sdk = BlackbirdSDK()
    
    # Act
    result = sdk.some_method()
    
    # Assert
    assert result is not None
    assert result.status == "success"
```

## 🔍 Code Review Process

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] Examples work correctly
- [ ] No breaking changes (or documented)
- [ ] Security considerations addressed

### Review Guidelines

- Be constructive and respectful
- Focus on code quality and functionality
- Suggest improvements when possible
- Ask questions if something is unclear

## 🏷️ Release Process

### Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Examples work correctly
- [ ] Version number updated
- [ ] Changelog updated
- [ ] Release notes prepared

## 🆘 Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

### Resources

- [README.md](README.md) - Main documentation
- [STARTUP_GUIDE.md](STARTUP_GUIDE.md) - Setup and installation
- [Examples](examples/) - Code examples
- [API Documentation](docs/api/) - Detailed API reference

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:

- GitHub contributors list
- Release notes
- Documentation acknowledgments
- Community highlights

---

**Thank you for contributing to Blackbird Open Source SDK! 🚀** 