# Ariaska RL

**ARIASKA RL** is an advanced AI-driven Red Team/Blue Team simulation and training framework. It leverages reinforcement learning, multi-agent coordination, and GPT-powered reasoning to simulate cyber-attack and defense scenarios for research, training, and automation.

## Features

- **Red Team & Blue Team Agents**: Competing AI agents for offensive and defensive cyber operations.
- **Multi-Agent Coordination**: Agents communicate and coordinate using a shared memory and agent manager.
- **Reinforcement Learning**: Policy/value networks for adaptive learning and strategy optimization.
- **GPT Integration**: Natural language reasoning, command suggestion, and scenario generation.
- **Knowledge Base**: Import/export of commands, tools, and tactics from JSON, TXT, and PDF sources.
- **Visualization & Monitoring**: Rich dashboards, live stats, and training visualizations.
- **Scenario Management**: Custom scenario injection and memory replay for reproducible experiments.

## Project Structure

```
Ariaska_RL/
  core/
    agents/           # RedAgent, BlueAgent, ScoutAgent, etc.
    environment/      # CyberEnvironment simulation
    logic/            # Chain building, output interpretation, redundancy detection
    models/           # PolicyNet, ValueNet, neural network layers
    monitor/          # StatsMonitor, dashboards
    teach/            # TeachModule, GPT integration
    visualization/    # Training visualizer, status panels
    ...
  data/
    knowledge_sources/  # JSON/TXT/PDF knowledge base files
    ...
  logs/               # Training and audit logs
  scripts/            # Utility scripts
  debug.py            # Audit and sanity check tool
  main.py             # Main entry point
```

## Getting Started

### Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Rich](https://github.com/Textualize/rich)
- (Optional) Access to GPT models (e.g., via `sgpt` CLI)

Install dependencies:
```sh
pip install -r backup_requirements.txt
```

### Running the Simulation

To launch a training session or simulation:
```sh
python Ariaska_RL/main.py
```

To run a full audit and sanity check:
```sh
python Ariaska_RL/debug.py
```

### Customization

- Add new commands, tools, or scenarios in `data/knowledge_sources/`.
- Implement new agents in `core/agents/`.
- Adjust RL parameters in agent classes (e.g., `BlueAgent`, `RedAgent`).

## Key Components

- [`core/agents/blue_agent.py`](Ariaska_RL/core/agents/blue_agent.py): Blue Team defensive agent.
- [`core/agents/red_agent.py`](Ariaska_RL/core/agents/red_agent.py): Red Team offensive agent.
- [`core/teach/teach.py`](Ariaska_RL/core/teach/teach.py): TeachModule for GPT-powered knowledge injection.
- [`core/logic/chainbuilder.py`](Ariaska_RL/core/logic/chainbuilder.py): Attack chain construction and storage.
- [`core/monitor/stats_monitor.py`](Ariaska_RL/core/monitor/stats_monitor.py): Training stats and dashboards.
- [`core/visualization/training_visualizer.py`](Ariaska_RL/core/visualization/training_visualizer.py): Training progress visualization.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License

This project is for research and educational purposes only. See [LICENSE](LICENSE) for details.

---

*ARIASKA RL — AI-Driven Cybersecurity Simulation & Training Platform*
