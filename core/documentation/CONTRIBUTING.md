# Contributing to ARIASKA_RL

Thank you for your interest in contributing to ARIASKA_RL! This document provides guidelines and instructions for contributing to the project effectively.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Core Development Principles](#core-development-principles)
- [Contribution Workflow](#contribution-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Common Design Patterns](#common-design-patterns)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

## Code of Conduct

This project is meant to be inclusive and welcoming to all. Please be respectful and considerate in your interactions with other contributors.

## Getting Started

### Prerequisites

- Python 3.11+
- PyTorch
- Rich (for UI)
- Access to LLM API (OpenAI/Azure) for GPT functionality

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ARIASKA_RL.git
   ```

2. Install dependencies:
   ```bash
   cd ARIASKA_RL
   pip install -r requirements.txt
   ```

3. Run the fix_imports script to ensure all imports are correctly resolved:
   ```bash
   bash fix_imports.sh
   ```

4. Configure your environment:
   ```bash
   cp .env.example .env  # Then edit with your API keys
   ```

5. Run initial verification:
   ```bash
   python debug.py --verify
   ```

## Core Development Principles

When contributing to ARIASKA_RL, please adhere to these core principles:

### 1. All LLM Calls Through GPTManager

All GPT/LLM interactions MUST go through the `GPTManager` class. This ensures:
- Consistent token tracking
- Proper caching and fallback
- Unified logging and monitoring
- Centralized security controls

```python
# ❌ BAD: Direct subprocess/API call
result = subprocess.run(["sgpt", prompt], capture_output=True, text=True)

# ✅ GOOD: Using GPTManager
from core.gpt_manager import GPTManager
gpt_manager = GPTManager()
result = gpt_manager.gpt_request(prompt, task_type="reasoning", agent_id="your_agent_id")
```

#### Examples of Different Task Types

```python
# For deep reasoning (uses primary model)
response = gpt_manager.gpt_request(
    prompt="Analyze this complex attack pattern...", 
    task_type="reasoning",
    agent_id="RedAgent"
)

# For lightweight analysis (uses faster model)
response = gpt_manager.gpt_request(
    prompt="Categorize this command output...", 
    task_type="analysis",
    agent_id="ScoutAgent"
)

# For embedding/vectorizing (uses nano model)
response = gpt_manager.gpt_request(
    prompt="Vectorize this context...", 
    task_type="embedding",
    agent_id="OrionAgent"
)
```

### 2. Unified Memory Schema

All agents should follow the unified memory schema:

```python
self.memory = {
    "actions": [],  # List of action records
    "rewards": {},  # Phase-keyed rewards
    "scenarios": [] # Special training scenarios
}
```

#### Action Record Structure

```python
action_record = {
    "command": "nmap -sV 10.10.10.10",  # The command executed
    "phase": "recon",                   # Attack phase
    "timestamp": 1718265896.245,        # Unix timestamp
    "reward": 12.5,                     # Reward received
    "output": "Port 22 open...",        # Command output
    "success": True,                    # Success flag
    "reasoning": "Identifies open services", # GPT reasoning
    "gpt_tokens": 320                   # Tokens used
}
```

### 3. Use MemoryRouter for Agent Transitions

All agent transitions should be logged through `MemoryRouter`:

```python
# Log transitions properly
self.memory_router.log_transition(
    self.agent_id, 
    state, 
    action, 
    reward, 
    next_state, 
    priority=abs(reward)+0.01,
    gpt_tokens=gpt_tokens
)
```

#### Accessing Global Memory

```python
# Get another agent's memory (for coordination)
blue_memory = self.memory_router.get_memory("BlueAgent")

# Get global insights across all agents
global_insights = self.memory_router.get_global_insights()

# Check if a similar experience exists (prevent redundancy)
is_redundant = self.memory_router.check_similar_experience(
    agent_id=self.agent_id,
    action="nmap -sV 10.10.10.10",
    state=current_state,
    threshold=0.85
)
```

### 4. Proper Error Handling

Always use try/except blocks for all external API calls and file operations:

```python
try:
    # Risky operation
    result = operation()
except Exception as e:
    console.print(f"[red]❌ Error: {e}[/red]")
    # Provide fallback or graceful degradation
    result = fallback_operation()
```

#### Specific Exception Handling

```python
try:
    # External API call
    response = external_api_call()
except ConnectionError as e:
    console.print(f"[yellow]⚠ Connection error: {e}. Using cached response.[/yellow]")
    response = get_cached_response()
except TimeoutError as e:
    console.print(f"[yellow]⚠ Timeout error: {e}. Using simplified model.[/yellow]")
    response = fallback_model_call()
except Exception as e:
    console.print(f"[red]❌ Unexpected error: {e}[/red]")
    response = safe_default_response()
```

### 5. Interface Consistency

All agents must implement the appropriate interfaces:

```python
class YourAgent(AgentInterface, MemorySyncInterface):
    # Must implement all interface methods
    def act(self, state):
        # Implementation
    
    def learn(self):
        # Implementation
        
    def sync_memory(self):
        # Implementation
```

#### Example Interface Implementation

```python
from core.interfaces.agent_interface import AgentInterface
from core.interfaces.memory_sync_interface import MemorySyncInterface

class CustomAgent(AgentInterface, MemorySyncInterface):
    def __init__(self, agent_id="CustomAgent", memory_router=None):
        self.agent_id = agent_id
        self.memory_router = memory_router
    
    def act(self, state):
        """Required by AgentInterface"""
        # Decision logic here
        return "nmap -sV 10.10.10.10"
    
    def learn(self):
        """Required by AgentInterface"""
        # Learning logic here
        pass
    
    def sync_memory(self):
        """Required by MemorySyncInterface"""
        if self.memory_router:
            self.memory_router.save_memory(self.agent_id, self.memory)
```

## Contribution Workflow

1. **Fork the repository** and create a branch for your feature/fix
2. **Implement your changes**, following the development principles
3. **Run the fix_imports.sh** script to ensure imports are correct
4. **Add tests** for your changes when applicable
5. **Update documentation** as needed
6. **Submit a pull request** with a clear description of your changes

### Example Contribution Process

```bash
# Fork the repository through GitHub UI

# Clone your fork
git clone https://github.com/YOUR_USERNAME/ARIASKA_RL.git
cd ARIASKA_RL

# Create a feature branch
git checkout -b feature/improved-memory-management

# Make your changes to the codebase
# ...

# Run import fixer
bash fix_imports.sh

# Run tests
python -m unittest discover tests

# Commit your changes
git add .
git commit -m "Add improved memory management with deduplication"

# Push to your fork
git push origin feature/improved-memory-management

# Create a pull request through GitHub UI
```

## Code Style Guidelines

- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add docstrings to all functions and classes
- Keep functions focused on a single responsibility
- Use type hints where possible

### Method Docstring Format

```python
def method_name(self, param1, param2):
    """
    Brief description of method's purpose.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
        
    Returns:
        return_type: Description of return value
        
    Raises:
        ErrorType: When/why this error is raised
    """
```

### Function/Class Naming Conventions

- **Classes**: CamelCase (`RedAgent`, `MemoryRouter`)
- **Functions/Methods**: snake_case (`calculate_reward`, `get_phase_vector`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_EPISODES`, `DEFAULT_GAMMA`)
- **Private Methods**: prefixed with underscore (`_sanitize_output`)

## Documentation Guidelines

- Document all public methods with docstrings
- Keep the README.md up to date
- Add inline comments for complex logic
- Update ARCHITECTURE.md when making structural changes
- Create diagrams for complex workflows using Mermaid

### Rich UI Guidelines

For terminal UI components using Rich:

- Use consistent colors for similar information
- Include progress indicators for long operations
- Ensure error messages are clearly highlighted
- Use panels to group related information

#### Color Conventions

- **Green** (`[green]text[/green]`): Success messages, positive rewards
- **Red** (`[red]text[/red]`): Errors, failures, negative rewards
- **Yellow** (`[yellow]text[/yellow]`): Warnings, cautions, mixed outcomes
- **Cyan** (`[cyan]text[/cyan]`): Information, phase names, agent IDs
- **Magenta** (`[magenta]text[/magenta]`): GPT responses, strategic insights
- **Blue** (`[blue]text[/blue]`): System messages, environment states

## Testing Guidelines

- Write unit tests for new functionality
- Test edge cases and failure modes
- Ensure compatibility with both simulated and live environments
- Verify GPT fallback paths work correctly when API is unavailable
- Test with GPT caching enabled and disabled

### Example Test Structure

```python
import unittest
from unittest.mock import MagicMock, patch
from core.agents.scout_agent import ScoutAgent

class TestScoutAgent(unittest.TestCase):
    def setUp(self):
        self.memory_router = MagicMock()
        self.agent = ScoutAgent(memory_router=self.memory_router)
        
    def test_advise_phase_with_valid_state(self):
        # Arrange
        state = {"open_ports": [22, 80], "privilege_level": "none"}
        
        # Act
        phase = self.agent.advise_phase(state)
        
        # Assert
        self.assertIn(phase, ["recon", "enumeration", "exploit", "privesc", "exfiltrate"])
        
    @patch('core.gpt_manager.GPTManager.gpt_request')
    def test_advise_phase_with_gpt_error(self, mock_gpt):
        # Arrange
        mock_gpt.side_effect = Exception("GPT unavailable")
        state = {"open_ports": [22, 80], "privilege_level": "none"}
        
        # Act
        phase = self.agent.advise_phase(state)
        
        # Assert
        self.assertEqual(phase, "recon")  # Default fallback
```

## Common Design Patterns

The ARIASKA_RL project uses several design patterns consistently:

### 1. Factory Pattern

```python
# In core/multiagent/agents.py
def create_agent(agent_type, **kwargs):
    if agent_type == "RedAgent":
        from core.agents.red_agent import RedAgent
        return RedAgent(**kwargs)
    elif agent_type == "BlueAgent":
        from core.agents.blue_agent import BlueAgent
        return BlueAgent(**kwargs)
    # ...
```

### 2. Observer Pattern

```python
# Using the broadcast/query pattern in AgentManager
# Broadcaster
self.agent_manager.broadcast(
    f"{self.agent_id}_phase",
    state.get("phase", "N/A"),
    sender=self.agent_id,
)

# Observer
current_red_phase = self.agent_manager.query_context("RedAgent_phase")
```

### 3. Strategy Pattern

```python
# Different strategies for command selection
def select_command(self, state, strategy="balanced"):
    if strategy == "stealthy":
        return self._select_stealthy_command(state)
    elif strategy == "aggressive":
        return self._select_aggressive_command(state)
    else:
        return self._select_balanced_command(state)
```

### 4. Singleton Pattern

```python
# For shared managers that should exist once
class GPTManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance
        
    def _initialize(self, *args, **kwargs):
        # Actual initialization code
```

## Advanced Topics

### Adding a New Agent

1. Create a new class in `core/agents/` that implements the required interfaces
2. Register the agent in `core/multiagent/agents.py`
3. Update `AgentManager` to handle the new agent type
4. Add appropriate memory management in `MemoryRouter`

```python
# 1. Create agent class in core/agents/your_agent.py
from core.interfaces.agent_interface import AgentInterface
from core.interfaces.memory_sync_interface import MemorySyncInterface

class YourAgent(AgentInterface, MemorySyncInterface):
    def __init__(self, agent_id="YourAgent", memory_router=None):
        self.agent_id = agent_id
        self.memory_router = memory_router
        self.memory = {"actions": [], "rewards": {}, "scenarios": []}
        
    # Implement required methods
    
# 2. Register in core/multiagent/agents.py
def get_all_agents(agent_manager=None, memory_router=None):
    agents = {
        # Existing agents...
        "YourAgent": YourAgent(
            agent_manager=agent_manager,
            memory_router=memory_router
        )
    }
    return agents

# 3. Update AgentManager initialization
def _initialize_agents(self):
    # ...existing code...
    self.your_agent = agents["YourAgent"]
    self.agents.append(self.your_agent)
```

### Adding New Environment Types

1. Extend the `CyberEnvironment` class in `core/environment/`
2. Update `EnvironmentContextDetector` to recognize the new environment
3. Add any necessary safety controls for the environment
4. Document environment parameters in the config

```python
# 1. Extend CyberEnvironment
class DockerizedEnvironment(CyberEnvironment):
    def __init__(self, container_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.container_name = container_name
        
    def reset(self):
        # Custom reset logic for Docker environment
        
    def step(self, action):
        # Custom step implementation for Docker
```

### Adding New LLM/GPT Models

1. Extend the `GPTManager` class to support the new model
2. Add appropriate caching and fallback logic
3. Update model selection logic based on task type
4. Add token tracking for the new model

```python
class GPTManager:
    def __init__(self):
        # Add new model to the available models
        self.models = {
            "gpt-4o-mini": {"priority": 1, "cost_per_token": 0.001},
            "gpt-4.1": {"priority": 2, "cost_per_token": 0.002},
            "new-model": {"priority": 3, "cost_per_token": 0.003}
        }
        
    def _select_model(self, task_type):
        # Update model selection logic
        if task_type == "critical_reasoning":
            return "new-model"  # Use new model for critical tasks
        # ...existing logic
```

## Troubleshooting

### Common Issues and Solutions

#### Import Errors

**Issue**: `ImportError: cannot import name X from Y`
**Solution**: Run the fix_imports.sh script which resolves common import path issues.

```bash
bash fix_imports.sh
```

#### GPT Token Limits

**Issue**: Running out of GPT tokens during operation
**Solution**: Configure token limits and fallback models in .env

```
GPT_TOKEN_LIMIT=10000
GPT_PRIMARY_MODEL=gpt-4o-mini
GPT_FALLBACK_MODEL=gpt-4.1-nano
```

#### Memory Growth Issues

**Issue**: Memory usage growing too large during long episodes
**Solution**: Enable automatic pruning in MemoryRouter and adjust ReplayBuffer capacity

```python
# In your agent initialization
self.replay_buffer = ReplayBuffer(
    capacity=1000,  # Smaller capacity
    use_sqlite=True  # Use DB-backed storage
)
```

#### Circular Import Problems

**Issue**: Circular imports causing module loading issues
**Solution**: Use dynamic imports within methods rather than at module level

```python
# Instead of top-level import
def method_that_needs_module():
    from core.some_module import SomeClass
    # Use SomeClass here
```

## Questions and Support

If you have questions or need help, please:
1. Check existing documentation
2. Look for related issues in the issue tracker
3. Open a new issue with the "question" label if needed
4. Join the community discussions in Discord

Thank you for contributing to ARIASKA_RL! Your efforts help us build a better, more capable multi-agent cybersecurity platform.
