#!/usr/bin/env python3
"""
ARIASKA_RL - Standalone Training System
Unified entry point for training intelligent cybersecurity agents.
Can be run directly: python training.py <episodes>
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.training.enhanced_unified_trainer import EnhancedUnifiedTrainer


def setup_logging() -> logging.Logger:
    """Setup logging for training"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("ARIASKA_TRAINING")


def validate_environment() -> bool:
    """Validate that the environment is properly set up"""
    logger = logging.getLogger("ARIASKA_TRAINING")
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        logger.error("❌ .env file not found. Please create one based on .env.example")
        return False
    
    # Check for core modules
    core_modules = [
        "core/training/unified_trainer.py",
        "core/gpt_manager.py"
    ]
    
    for module in core_modules:
        if not Path(module).exists():
            logger.error(f"❌ Required module not found: {module}")
            return False
    
    logger.info("✅ Environment validation passed")
    return True


def main():
    """Main training entry point"""
    # Setup logging
    logger = setup_logging()
    
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python training.py <episodes>")
        print("Example: python training.py 5")
        sys.exit(1)
    
    try:
        episodes = int(sys.argv[1])
        if episodes <= 0 or episodes > 1000:
            raise ValueError("Episodes must be between 1 and 1000")
    except ValueError as e:
        logger.error(f"❌ Invalid episodes value: {e}")
        sys.exit(1)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Initialize and run training
    try:
        logger.info(f"🚀 Starting ARIASKA_RL Training Session")
        logger.info(f"📊 Episodes: {episodes}")
        logger.info(f"🎯 Mode: Advanced Deep RL + GPT-4o-mini")
        
        # Create trainer
        trainer = EnhancedUnifiedTrainer(episodes=episodes)
        
        # Run training
        start_time = time.time()
        results = trainer.train()
        duration = time.time() - start_time
        
        # Log results
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"⏱️  Duration: {duration:.2f} seconds")
        logger.info(f"📈 Final accuracy: {results.get('performance', {}).get('average_accuracy', 0):.1%}")
        
        # Save summary
        summary_file = Path("logs") / f"training_summary_{int(time.time())}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'episodes': episodes,
                'duration': duration,
                'results': results,
                'timestamp': int(time.time())
            }, f, indent=2)
        
        logger.info(f"📄 Training summary saved to: {summary_file}")
        
    except KeyboardInterrupt:
        logger.warning("🛑 Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
