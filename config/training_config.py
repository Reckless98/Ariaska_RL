# Enhanced Training Configuration for ARIASKA_RL v2.0
# Optimized for GPT-4o-mini integration and continuous learning

# Neural Network Architecture Configuration
POLICY_NET_CONFIG = {
    "input_size": 512,
    "hidden_sizes": [256, 256, 128],  # Enhanced depth
    "output_size": 5,
    "activation": "gelu",
    "dropout": 0.1,
    "layer_norm": True,
    "dueling": True,
    "noisy_layers": True
}

VALUE_NET_CONFIG = {
    "input_size": 512,
    "hidden_sizes": [384, 256, 128],  # Enhanced depth
    "activation": "gelu",
    "dropout": 0.15,
    "layer_norm": True,
    "batch_norm": False,
    "residual_connections": True
}

# Training Hyperparameters
TRAINING_CONFIG = {
    # Learning rates
    "policy_lr": 1e-4,
    "value_lr": 1e-4,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 100,
    
    # Training dynamics
    "batch_size": 64,
    "replay_buffer_size": 10000,
    "target_update_frequency": 1000,
    "gradient_clipping": 1.0,
    
    # Exploration
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "entropy_beta": 0.01,
    
    # Experience replay
    "prioritized_replay": True,
    "alpha": 0.6,
    "beta": 0.4,
    "beta_increment": 0.001,
    
    # Multi-step learning
    "n_step": 3,
    "gamma": 0.99,
    
    # GPT integration
    "gpt_feedback_frequency": 10,  # Every 10 episodes
    "gpt_guided_exploration": True,
    "learning_insights_window": 50,
    
    # Memory optimization
    "memory_cleanup_frequency": 100,
    "memory_compression": True,
    "cross_agent_memory_sync": True
}

# Agent-specific configurations
AGENT_CONFIGS = {
    "RedAgent": {
        "specialization": "offensive",
        "exploration_bonus": 0.1,
        "risk_tolerance": 0.8,
        "gpt_complexity": "tactical"
    },
    "BlueAgent": {
        "specialization": "defensive", 
        "exploration_bonus": 0.05,
        "risk_tolerance": 0.3,
        "gpt_complexity": "strategic"
    },
    "ScoutAgent": {
        "specialization": "reconnaissance",
        "exploration_bonus": 0.2,
        "risk_tolerance": 0.6,
        "gpt_complexity": "tactical"
    },
    "ShadowAgent": {
        "specialization": "stealth",
        "exploration_bonus": 0.15,
        "risk_tolerance": 0.7,
        "gpt_complexity": "strategic"
    },
    "OrionAgent": {
        "specialization": "coordination",
        "exploration_bonus": 0.0,
        "risk_tolerance": 0.4,
        "gpt_complexity": "advanced"
    }
}

# Performance optimization
OPTIMIZATION_CONFIG = {
    "mixed_precision": True,
    "gradient_accumulation": 2,
    "dataloader_workers": 2,
    "pin_memory": True,
    "async_env_steps": True,
    "vectorized_environments": 4
}
