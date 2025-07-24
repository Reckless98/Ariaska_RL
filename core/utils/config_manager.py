"""
Configuration management system for ARIASKA_RL.
Provides centralized, type-safe configuration handling.
"""
import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    learning_rate: float = 1e-4
    batch_size: int = 64
    memory_size: int = 100000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1000
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = 'relu'
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    dueling: bool = True
    target_update_frequency: int = 100
    gradient_clip_norm: float = 1.0


@dataclass
class EnvironmentConfig:
    """Configuration for the cyber environment"""
    scenario: str = "dynamic"
    mode: str = "simulated"  # simulated, live, dynamic
    max_steps: int = 1000
    reward_scale: float = 1.0
    action_space_size: int = 4
    state_space_size: int = 128
    network_range: str = "192.168.1.0/24"
    scan_timeout: int = 30
    max_concurrent_scans: int = 5
    safe_mode: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    episodes: int = 1000
    max_steps_per_episode: int = 1000
    save_frequency: int = 100
    log_frequency: int = 10
    evaluation_frequency: int = 50
    evaluation_episodes: int = 10
    early_stopping_patience: int = 100
    target_reward: Optional[float] = None
    curriculum_learning: bool = False


@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    max_tokens: int = 2048
    temperature: float = 0.7
    max_retries: int = 3
    request_timeout: int = 30
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    fallback_model: str = "gpt-3.5-turbo"


@dataclass
class VisualizationConfig:
    """Configuration for visualization and monitoring"""
    enable_dashboard: bool = True
    dashboard_update_frequency: int = 10
    save_plots: bool = True
    plot_format: str = "png"
    tensorboard_enabled: bool = True
    tensorboard_log_dir: str = "runs"
    wandb_enabled: bool = False
    wandb_project: str = "ariaska_rl"
    wandb_entity: str = ""


@dataclass
class SystemConfig:
    """System-level configuration"""
    device: str = "auto"  # auto, cpu, cuda, cuda:0, etc.
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = False
    distributed_training: bool = False
    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 12355


@dataclass
class SecurityConfig:
    """Security and safety configuration"""
    safe_mode: bool = True
    validate_inputs: bool = True
    sanitize_outputs: bool = True
    max_api_calls_per_minute: int = 60
    allowed_ip_ranges: List[str] = field(default_factory=lambda: ["192.168.0.0/16", "10.0.0.0/8"])
    log_sensitive_data: bool = False


@dataclass
class StorageConfig:
    """Storage and persistence configuration"""
    checkpoint_dir: str = "data/checkpoints"
    experiment_name: str = "ariaska_experiment"
    data_dir: str = "data"
    logs_dir: str = "logs"
    chroma_persist_directory: str = "chroma_data"
    sqlite_db_path: str = "data/ariaska.db"
    auto_cleanup: bool = True
    max_checkpoints: int = 10


@dataclass
class Config:
    """Main configuration class combining all sub-configurations"""
    agent: AgentConfig = field(default_factory=AgentConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._validate_config()
        self._setup_device()
        self._create_directories()
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate learning rate
        if self.agent.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate batch size
        if self.agent.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Validate epsilon values
        if not (0 <= self.agent.epsilon_end <= self.agent.epsilon_start <= 1):
            raise ValueError("Epsilon values must be between 0 and 1, with epsilon_end <= epsilon_start")
        
        # Validate environment settings
        if self.environment.max_steps <= 0:
            raise ValueError("Max steps must be positive")
        
        # Validate training settings
        if self.training.episodes <= 0:
            raise ValueError("Number of episodes must be positive")
    
    def _setup_device(self):
        """Setup compute device"""
        if self.system.device == "auto":
            if torch.cuda.is_available():
                self.system.device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.system.device = "cpu"
                logger.info("Using CPU")
        else:
            logger.info(f"Using specified device: {self.system.device}")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.storage.checkpoint_dir,
            self.storage.data_dir,
            self.storage.logs_dir,
            self.storage.chroma_persist_directory,
            os.path.dirname(self.storage.sqlite_db_path),
        ]
        
        if self.visualization.tensorboard_enabled:
            directories.append(self.visualization.tensorboard_log_dir)
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def save(self, filepath: str):
        """Save configuration to YAML file"""
        config_dict = self._to_dict()
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'agent': self.agent.__dict__,
            'environment': self.environment.__dict__,
            'training': self.training.__dict__,
            'llm': {k: v for k, v in self.llm.__dict__.items() if k != 'api_key'},  # Don't save API key
            'visualization': self.visualization.__dict__,
            'system': self.system.__dict__,
            'security': self.security.__dict__,
            'storage': self.storage.__dict__,
        }


class ConfigManager:
    """Configuration manager for loading and managing configurations"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/default.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Config:
        """Load configuration from file and environment variables"""
        # Start with default configuration
        config = Config()
        
        # Load from YAML file if it exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Update configuration with YAML values
            config = self._update_config_from_dict(config, yaml_config)
        
        # Override with environment variables
        config = self._update_config_from_env(config)
        
        return config
    
    def _update_config_from_dict(self, config: Config, config_dict: Dict[str, Any]) -> Config:
        """Update configuration from dictionary"""
        for section_name, section_dict in config_dict.items():
            if hasattr(config, section_name) and isinstance(section_dict, dict):
                section = getattr(config, section_name)
                for key, value in section_dict.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        
        return config
    
    def _update_config_from_env(self, config: Config) -> Config:
        """Update configuration from environment variables"""
        env_mappings = {
            # LLM configuration
            'OPENAI_API_KEY': ('llm', 'api_key'),
            'OPENAI_MODEL': ('llm', 'model'),
            'OPENAI_MAX_TOKENS': ('llm', 'max_tokens'),
            
            # Environment configuration
            'ENVIRONMENT_MODE': ('environment', 'mode'),
            'SCENARIO': ('environment', 'scenario'),
            'MAX_STEPS_PER_EPISODE': ('environment', 'max_steps'),
            
            # Training configuration
            'TRAINING_EPISODES': ('training', 'episodes'),
            'BATCH_SIZE': ('agent', 'batch_size'),
            'LEARNING_RATE': ('agent', 'learning_rate'),
            
            # System configuration
            'CUDA_VISIBLE_DEVICES': ('system', 'device'),
            'NUM_WORKERS': ('system', 'num_workers'),
            
            # Visualization
            'ENABLE_DASHBOARD': ('visualization', 'enable_dashboard'),
            'TENSORBOARD_LOG_DIR': ('visualization', 'tensorboard_log_dir'),
            
            # Storage
            'CHECKPOINT_DIR': ('storage', 'checkpoint_dir'),
            'EXPERIMENT_NAME': ('storage', 'experiment_name'),
        }
        
        for env_var, (section_name, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section = getattr(config, section_name)
                
                # Convert string values to appropriate types
                if hasattr(section, key):
                    current_value = getattr(section, key)
                    if isinstance(current_value, bool):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                    elif isinstance(current_value, list):
                        value = [item.strip() for item in value.split(',')]
                    
                    setattr(section, key, value)
        
        return config
    
    def get_config(self) -> Config:
        """Get the current configuration"""
        return self.config
    
    def reload_config(self) -> Config:
        """Reload configuration from file"""
        self.config = self._load_config()
        return self.config
    
    def validate_config(self) -> tuple[bool, List[str]]:
        """Validate the current configuration"""
        errors = []
        
        try:
            # Re-run validation
            self.config._validate_config()
        except ValueError as e:
            errors.append(str(e))
        
        # Additional validations
        if self.config.llm.api_key == "":
            errors.append("OpenAI API key is not set")
        
        if not os.path.exists(self.config.storage.data_dir):
            errors.append(f"Data directory does not exist: {self.config.storage.data_dir}")
        
        return len(errors) == 0, errors


# Global configuration manager instance
_config_manager = None

def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance"""
    global _config_manager
    
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager.get_config()


def reload_config() -> Config:
    """Reload global configuration"""
    global _config_manager
    
    if _config_manager is not None:
        return _config_manager.reload_config()
    else:
        return get_config()


# Example usage and defaults
if __name__ == "__main__":
    # Create default configuration
    config = Config()
    
    # Save default configuration as example
    os.makedirs("config", exist_ok=True)
    config.save("config/default.yaml")
    
    print("Default configuration saved to config/default.yaml")
    
    # Validate configuration
    manager = ConfigManager()
    is_valid, errors = manager.validate_config()
    
    if is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")