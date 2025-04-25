# ARIASKA_RL — Autonomous Red Team RL Framework

## Key Features
- **RedAgent Online Self-Evolution**: RedAgent logs every turn (state, command, GPT response, output, metrics) into an episodic memory. After each episode, it uses its own logs as feedback, querying GPT-4.1 for analysis and strategy refinement. This enables continual, on-the-fly learning and adaptation.
- **SGPT Integration Loop**: All GPT calls are routed through a smart GPTManager, which dynamically chooses between GPT-4.1 and GPT-4o-mini based on token budget and latency. Prompt-response caching avoids redundant API calls, and all GPT feedback is logged for future learning.
- **Visual Rich CLI Training UI**: A live Rich-based dashboard (`display_redagent_learning_dashboard`) shows RedAgent’s evolving strategies, command timeline, GPT call log, memory snapshot, and reward/command heatmap. This dashboard updates after every RedAgent step.
- **Meta-Learning Loop**: Every N episodes, RedAgent’s performance is summarized and sent to GPT-4.1 for meta-analysis. GPT’s recommendations are logged and can influence future exploration/exploitation.
- **MemoryRouter Evolution Log**: All RedAgent steps are also logged to a global evolution log for deduplication and cross-episode statistics (command success rates, most/least effective commands, etc.).
- **Best Practices**: All memory writes are append-only and rate-limited. Policy/strategy updates only occur between episodes. GPT failures are handled gracefully with fallback and caching.

## How It Works
1. **RedAgent Step**: After each action, RedAgent logs the step to its episodic memory and the global evolution log, then calls GPTManager for feedback.
2. **Learning from Feedback**: At the end of each episode, RedAgent analyzes its recent steps with GPT-4.1, logs improvements, and updates its internal strategy.
3. **Meta-Learning**: Every N episodes, RedAgent’s evolution stats are summarized and sent to GPT-4.1 for high-level recommendations.
4. **Live Dashboard**: The dashboard is updated after every RedAgent step, providing real-time insight into learning and strategy evolution.

## Usage
- Run `main.py` or `core/trainer.py` to start training. The dashboard and meta-learning loop are enabled by default.
- All GPT calls are cached and logged. You can review RedAgent’s evolution log in `core/memories/shared/redagent_evolution/evolution_log.jsonl`.
- For advanced visualization, use the Rich dashboard or the `core/visualization/training_visualizer.py`.

## Research Foundations
- Reflexion agents: Verbal feedback and memory for LLM improvement ([arxiv.org](https://arxiv.org))
- AGILE: Reflection, memory, and reinforcement in LLM agents ([arxiv.org](https://arxiv.org))
- Dynamic model routing and prompt caching ([medium.com](https://medium.com))
- Rich live display for RL ([realpython.com](https://realpython.com))

## Project Structure
- `core/agents/red_agent.py`: RedAgent logic, learning loop, GPTManager integration
- `core/agents/redagent_brain.py`: Episodic memory, feedback logging, learning/reflection
- `core/multiagent/memory_router.py`: Global evolution log, stats, deduplication
- `core/ui_helpers.py`: Rich dashboard for RedAgent learning
- `core/trainer.py`, `main.py`: Training loop, meta-learning, dashboard integration

## Best Practices
- All learning is safe and structured. No policy updates mid-episode.
- Memory is append-only and periodically flushed to disk.
- GPT-4.1 is used for deep reasoning, GPT-4o-mini for quick checks and fallback.
- All failures are handled gracefully with fallback and caching.

## Features

- **Red Team & Blue Team Agents**: Competing AI agents for offensive and defensive cyber operations.
- **Multi-Agent Coordination**: Agents communicate and coordinate using a shared memory and agent manager.
- **Reinforcement Learning**: Policy/value networks for adaptive learning and strategy optimization.
- **GPT Integration**: Natural language reasoning, command suggestion, and scenario generation.
- **Knowledge Base**: Import/export of commands, tools, and tactics from JSON, TXT, and PDF sources.
- **Visualization & Monitoring**: Rich dashboards, live stats, and training visualizations.
- **Scenario Management**: Custom scenario injection and memory replay for reproducible experiments.

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

## Developer Onboarding & Extensibility

### LLM Orchestration: GPTManager
- All GPT/LLM calls must be routed through `core/gpt_manager.py` (`GPTManager`).
- `GPTManager` handles model selection, prompt caching, token tracking, and fallback logic.
- To add a new LLM provider (e.g., Anthropic, local models), extend `GPTManager` with a new backend and update the model selection logic.
- Never call GPT APIs or subprocesses directly from agents/utilities—always use `GPTManager`.

### Memory & Coordination: MemoryRouter
- All agent actions, transitions, and GPT interactions are logged via `core/memory_router.py` (`MemoryRouter`).
- `MemoryRouter` provides deduplication, prioritized replay, and global insight synchronization.
- To extend memory or add new logging features, subclass or update `MemoryRouter`.

### Extending Agents & Environments
- New agents should inherit from `AgentInterface` and use `MemoryRouter` and `GPTManager` for all memory and LLM operations.
- To add a new environment (simulated or live), implement a new handler in `core/environment/` and register it with the environment manager.
- For plugin-style expansion, use dynamic imports and configuration files to register new agent/environment types.

### Security & Compliance
- All LLM outputs must be sanitized before execution or logging.
- Never allow dynamic code execution from LLMs without strict validation.
- Audit logs and memory for sensitive data and redact as needed.

### Monitoring & Deployment
- Prometheus/Grafana/Streamlit integration points are scaffolded for advanced monitoring.
- Use the provided `Dockerfile` and `docker-compose.yml` for modular, containerized deployment.
- All configs and secrets should be managed via environment variables or config files.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License

This project is for research and educational purposes only. See [LICENSE](LICENSE) for details.

---

*ARIASKA RL — AI-Driven Cybersecurity Simulation & Training Platform*