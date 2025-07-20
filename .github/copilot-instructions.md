# GitHub Copilot Instructions for Ariaska_RL

## Project Overview
This is a Reinforcement Learning (RL) project named Ariaska_RL. The project aims to implement and experiment with various RL algorithms and environments.

## Code Standards and Best Practices

### Python Standards
- Use Python 3.8+ features and type hints
- Follow PEP 8 style guide
- Use descriptive variable names (e.g., `reward_buffer` instead of `r_buf`)
- Add docstrings to all functions and classes using Google style
- Implement error handling with meaningful exception messages

### Project Structure
```
Ariaska_RL/
├── agents/           # RL agent implementations
├── environments/     # Custom environments
├── models/          # Neural network architectures
├── utils/           # Helper functions and utilities
├── configs/         # Configuration files
├── experiments/     # Experiment scripts and results
├── tests/           # Unit and integration tests
└── notebooks/       # Jupyter notebooks for exploration
```

### Dependencies
When suggesting code, assume these key libraries are available:
- `torch` (PyTorch) for neural networks
- `numpy` for numerical operations
- `gym` or `gymnasium` for RL environments
- `tensorboard` for logging
- `matplotlib` for visualization
- `tqdm` for progress bars

## RL-Specific Guidelines

### Agent Implementation
When implementing RL agents:
1. Create a base agent class with common methods
2. Implement specific algorithms (DQN, PPO, A3C, etc.) as subclasses
3. Include replay buffer functionality where applicable
4. Add proper exploration strategies (epsilon-greedy, noise-based, etc.)

Example structure:
```python
class BaseAgent:
    def __init__(self, state_dim, action_dim, config):
        pass
    
    def select_action(self, state, training=True):
        """Select action with exploration during training"""
        pass
    
    def update(self, batch):
        """Update agent parameters"""
        pass
    
    def save(self, path):
        """Save model checkpoint"""
        pass
```

### Environment Guidelines
- Support both discrete and continuous action spaces
- Implement proper reset and step methods
- Include reward shaping capabilities
- Add visualization methods for debugging

### Training Loop Pattern
```python
def train(agent, env, config):
    for episode in range(config.num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.buffer) > config.batch_size:
                agent.update()
            
            state = next_state
            episode_reward += reward
        
        # Log metrics
        log_metrics(episode, episode_reward)
```

## Configuration Management
- Use YAML or JSON for hyperparameter configuration
- Implement a config parser that validates parameters
- Support command-line overrides for experiments

## Testing Requirements
- Write unit tests for critical components
- Test environment step/reset functionality
- Verify agent learning on simple tasks
- Include integration tests for full training pipelines

## Documentation Standards
- Add README.md for each module explaining its purpose
- Include examples of how to use each component
- Document hyperparameter choices and their effects
- Maintain a CHANGELOG.md for version tracking

## Performance Optimization
- Use vectorized environments for parallel training
- Implement efficient replay buffers with circular arrays
- Add GPU support with proper device management
- Profile code to identify bottlenecks

## Experiment Tracking
- Log all hyperparameters and metrics
- Save model checkpoints periodically
- Create reproducible experiment scripts
- Generate plots for learning curves and policy visualization

## Error Handling
- Validate input dimensions for networks
- Check action/observation space compatibility
- Handle environment termination gracefully
- Add informative error messages for debugging

## Code Generation Preferences
When generating code:
1. Start with simple, working implementations
2. Add complexity incrementally
3. Include usage examples in docstrings
4. Suggest appropriate hyperparameters
5. Add TODO comments for future improvements

## Common Tasks to Implement
1. **Basic Agents**: DQN, DDPG, PPO, SAC
2. **Environments**: CartPole wrapper, custom grid worlds
3. **Utilities**: Replay buffer, normalizers, schedulers
4. **Visualization**: Policy visualization, value function plots
5. **Benchmarking**: Performance comparison scripts

## Security and Best Practices
- Never hardcode API keys or sensitive data
- Use environment variables for configuration
- Implement proper random seed management
- Add input validation for all public methods

When asked to implement features, provide complete, working code with proper error handling and documentation. Focus on modularity and reusability to build a robust RL framework.