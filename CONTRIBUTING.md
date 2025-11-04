# Contributing to NeuroResearch-AI

Thank you for your interest in contributing to NeuroResearch-AI!

## Ways to Contribute

### Bug Reports
- Use the GitHub issue tracker
- Include detailed reproduction steps
- Provide environment information (Python version, OS, etc.)
- Include relevant logs and error messages

### Feature Requests
- Describe the feature and its use case
- Explain how it fits with the project goals
- Consider implementation approaches

### Research Domain Extensions
- Add new research domain templates
- Implement domain-specific methodologies
- Contribute evaluation metrics

### Agent Enhancements
- Improve existing agent capabilities
- Add new specialized agents
- Enhance inter-agent collaboration

### Documentation
- Improve README and examples
- Add tutorials and guides
- Document API references

## Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/NeuroResearch-AI.git
   cd NeuroResearch-AI
   ```

2. **Create development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests**
   ```bash
   pytest
   ```

## Code Standards

### Style Guidelines
- Follow PEP 8 (enforced by Black)
- Use type hints for all functions
- Write comprehensive docstrings
- Keep functions focused and modular

### Testing Requirements
- Write tests for new features
- Maintain >90% test coverage
- Include both unit and integration tests
- Test async functionality properly

### Commit Messages
- Use conventional commit format
- Examples:
  - `feat: add neuroscience research template`
  - `fix: resolve agent coordination bug`
  - `docs: update API documentation`

## Pull Request Process

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Test locally**
   ```bash
   pytest
   black neuroresearch_ai/
   isort neuroresearch_ai/
   flake8 neuroresearch_ai/
   mypy neuroresearch_ai/
   ```

4. **Submit pull request**
   - Provide clear description
   - Link related issues
   - Include screenshots if applicable

5. **Code review**
   - Address reviewer feedback
   - Update tests and docs as needed
   - Ensure CI passes

## Architecture Guidelines

### Agent Design
- Each agent should have a clear, focused responsibility
- Use appropriate LLM models for agent capabilities
- Implement proper error handling and fallbacks
- Support both sync and async operations

### Research Workflow
- Design workflows to be modular and extensible
- Support dynamic routing based on research needs
- Implement quality checkpoints and feedback loops
- Enable meta-learning and continuous improvement

### Cross-Domain Integration
- Design for extensibility across research domains
- Support domain-specific customizations
- Enable knowledge transfer between domains
- Maintain domain expertise while allowing synthesis

## Priority Areas

We especially welcome contributions in these areas:

### High Priority
- **Ethics Guardian Enhancements**: Improve automated ethics verification
- **Meta-Learning Algorithms**: Enhance self-improvement capabilities
- **Cross-Domain Synthesis**: Better knowledge integration across domains
- **Performance Optimization**: Improve system speed and efficiency

### Medium Priority
- **Additional Research Domains**: Psychology, Chemistry, Biology templates
- **Advanced Quality Metrics**: More sophisticated research evaluation
- **Collaborative Features**: Multi-researcher project support
- **Visualization Tools**: Research progress and results visualization

### Low Priority
- **GUI Interface**: Web-based research management interface
- **Cloud Integration**: Support for cloud compute resources
- **Advanced Analytics**: Research trend analysis and predictions
- **Mobile Companion**: Mobile app for research monitoring

## Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `research-domain`: Related to specific research domains
- `agent-enhancement`: Improvements to AI agents
- `meta-learning`: Self-improvement capabilities
- `cross-domain`: Interdisciplinary features

## Community Guidelines

### Be Respectful
- Treat all contributors with respect
- Provide constructive feedback
- Welcome newcomers and help them get started

### Be Collaborative
- Share knowledge and expertise
- Help others learn and grow
- Build on each other's work

### Be Scientific
- Base decisions on evidence and research
- Cite sources and prior work
- Maintain scientific rigor

## Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Email**: sunghunkwag@gmail.com for direct communication

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Project documentation
- Conference presentations and papers

## License

By contributing to NeuroResearch-AI, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for helping advance scientific research through AI!**