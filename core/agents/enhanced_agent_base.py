#!/usr/bin/env python3
# core/agents/enhanced_agent_base.py — ARIASKA Enhanced Agent Base v2.0
# 🤖 Intelligent Agent Foundation | 🧠 GPT-4o-mini Integration | 🎯 Maximum Performance

import time
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import torch
import numpy as np

from core.gpt_manager import GPTManager
from core.models.advanced_networks import create_advanced_policy_network, create_advanced_value_network
from core.memory.enhanced_memory_sync import EnhancedMemorySync, MemoryInsight

logger = logging.getLogger("ariaska.enhanced_agent")

@dataclass
class AgentConfig:
    """Configuration for enhanced agents."""
    agent_id: str
    agent_type: str  # "red", "blue", "scout", "shadow", "orion"
    state_dim: int = 256
    action_dim: int = 64
    hidden_dims: Optional[List[int]] = None
    learning_rate: float = 3e-4
    batch_size: int = 64
    memory_size: int = 10000
    gpt_guidance: bool = True
    use_enhanced_training: bool = True
    sync_interval: float = 5.0
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 512, 256]

@dataclass
class ActionResult:
    """Result of an agent action."""
    action: Any
    success: bool
    reward: float
    observation: Dict[str, Any]
    info: Dict[str, Any]
    gpt_reasoning: Optional[str] = None

class EnhancedAgentBase(ABC):
    """
    Enhanced base class for all ARIASKA agents.
    
    Features:
    - GPT-4o-mini guided decision making
    - Advanced neural network architectures
    - Enhanced training system integration
    - Memory synchronization
    - Performance optimization
    - Real-time learning
    """
    
    def __init__(self, config: AgentConfig, memory_sync: Optional[EnhancedMemorySync] = None):
        self.config = config
        self.agent_id = config.agent_id
        self.agent_type = config.agent_type
        
        # Core components
        self.gpt_manager = GPTManager.get_instance()
        self.memory_sync = memory_sync
        
        # Neural networks
        hidden_dims = config.hidden_dims or [512, 512, 256]
        self.policy_network = create_advanced_policy_network(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=hidden_dims
        )
        
        self.value_network = create_advanced_value_network(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=hidden_dims
        )
        
        # Enhanced training system (simplified for now)
        # Training system placeholder (to be implemented)
        self.trainer = None
        if config.use_enhanced_training:
            # TODO: Implement enhanced training system
            logger.info(f"Enhanced training requested for {self.agent_id} - placeholder active")
        # Agent state
        self.current_state = None
        self.current_context = "initialization"
        self.action_history = []
        self.reward_history = []
        self.performance_metrics = {
            "total_actions": 0,
            "successful_actions": 0,
            "total_reward": 0.0,
            "avg_reward": 0.0,
            "learning_progress": 0.0
        }
        
        # Memory and insights
        self.recent_insights = []
        self.shared_knowledge = {}
        self.last_sync_time = time.time()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        
        logger.info(f"🤖 Enhanced {config.agent_type} agent '{config.agent_id}' initialized")
    
    @abstractmethod
    async def perceive_environment(self) -> Dict[str, Any]:
        """Perceive and analyze the current environment state."""
        pass
    
    @abstractmethod
    async def plan_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the next action based on current state."""
        pass
    
    @abstractmethod
    async def execute_action(self, action_plan: Dict[str, Any]) -> ActionResult:
        """Execute the planned action."""
        pass
    
    async def run_cycle(self) -> ActionResult:
        """
        Run a complete agent cycle: perceive -> plan -> execute -> learn.
        
        Returns:
            ActionResult: Result of the action execution
        """
        cycle_start = time.time()
        
        try:
            # 1. Perceive environment
            state = await self.perceive_environment()
            self.current_state = state
            
            # 2. Get GPT guidance if enabled
            gpt_reasoning = None
            if self.config.gpt_guidance:
                gpt_reasoning = await self._get_gpt_guidance(state)
            
            # 3. Plan action (combines neural network and GPT insights)
            action_plan = await self.plan_action(state)
            if gpt_reasoning:
                action_plan["gpt_reasoning"] = gpt_reasoning
            
            # 4. Execute action
            result = await self.execute_action(action_plan)
            result.gpt_reasoning = gpt_reasoning
            
            # 5. Learn from experience
            if self.trainer:
                await self._learn_from_experience(state, action_plan, result)
            
            # 6. Update metrics and memory
            self._update_metrics(result)
            await self._update_memory(state, action_plan, result)
            
            # 7. Sync with other agents
            if self.memory_sync and time.time() - self.last_sync_time > self.config.sync_interval:
                await self._sync_with_agents()
                self.last_sync_time = time.time()
            
            cycle_duration = time.time() - cycle_start
            logger.debug(f"🔄 {self.agent_id} cycle completed in {cycle_duration:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Agent cycle failed for {self.agent_id}: {e}")
            return ActionResult(
                action=None,
                success=False,
                reward=-1.0,
                observation={},
                info={"error": str(e)}
            )
    
    async def _get_gpt_guidance(self, state: Dict[str, Any]) -> str:
        """Get guidance from GPT-4o-mini for current situation."""
        try:
            # Prepare context for GPT
            context = self._prepare_gpt_context(state)
            
            prompt = f"""
            As an expert {self.agent_type} agent in cybersecurity operations, analyze the current situation and provide strategic guidance.
            
            Current State:
            {json.dumps(context, indent=2)}
            
            Agent Type: {self.agent_type}
            Recent Performance: {self.performance_metrics['avg_reward']:.3f} avg reward
            
            Provide:
            1. Situation analysis
            2. Recommended action approach
            3. Risk assessment
            4. Success probability estimation
            
            Focus on actionable intelligence and tactical precision.
            """
            
            response = self.gpt_manager.gpt_request(
                prompt,
                task_type="tactical_analysis",
                agent_id=self.agent_id,
                max_tokens=400
            )
            
            return response if response else "No GPT guidance available"
            
        except Exception as e:
            logger.warning(f"GPT guidance failed for {self.agent_id}: {e}")
            return "GPT guidance unavailable"
    
    def _prepare_gpt_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context information for GPT analysis."""
        context = {
            "current_state": state,
            "agent_context": self.current_context,
            "recent_actions": self.action_history[-5:],  # Last 5 actions
            "recent_rewards": self.reward_history[-5:],  # Last 5 rewards
            "performance": self.performance_metrics
        }
        
        # Add shared knowledge if available
        if self.shared_knowledge:
            context["shared_intelligence"] = self.shared_knowledge
        
        return context
    
    async def _learn_from_experience(self, state: Dict[str, Any], 
                                   action_plan: Dict[str, Any], 
                                   result: ActionResult):
        """Learn from the experience using enhanced training."""
        if not self.trainer:
            return
        
        try:
            # Prepare training data
            state_tensor = self._state_to_tensor(state)
            action_tensor = self._action_to_tensor(action_plan)
            reward = torch.tensor(result.reward, dtype=torch.float32, device=self.device)
            next_state_tensor = self._state_to_tensor(result.observation)
            done = not result.success
            
            # Store experience
            experience = {
                "state": state_tensor,
                "action": action_tensor,
                "reward": reward,
                "next_state": next_state_tensor,
                "done": done,
                "gpt_guidance": result.gpt_reasoning
            }
            
            # Store experience for future training
            # TODO: Integrate with enhanced trainer once interface is finalized
            
            # Update learning progress (simplified)
            self.performance_metrics["learning_progress"] = min(1.0, 
                self.performance_metrics["total_actions"] / 1000.0)
            
        except Exception as e:
            logger.warning(f"Learning failed for {self.agent_id}: {e}")
    
    def _state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """Convert state dictionary to tensor representation."""
        # Simple encoding for now - can be enhanced with more sophisticated methods
        features = []
        
        # Extract numeric features
        for key, value in state.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Simple string encoding (can be enhanced with embeddings)
                features.append(float(len(value) % 100))
            elif isinstance(value, list):
                features.extend([float(x) if isinstance(x, (int, float)) else 0.0 for x in value[:10]])
        
        # Pad or truncate to state_dim
        while len(features) < self.config.state_dim:
            features.append(0.0)
        features = features[:self.config.state_dim]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def _action_to_tensor(self, action_plan: Dict[str, Any]) -> torch.Tensor:
        """Convert action plan to tensor representation."""
        # Simple encoding for now
        features = []
        
        for key, value in action_plan.items():
            if key == "gpt_reasoning":
                continue
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                features.append(float(len(value) % 100))
        
        # Pad or truncate to action_dim
        while len(features) < self.config.action_dim:
            features.append(0.0)
        features = features[:self.config.action_dim]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def _update_metrics(self, result: ActionResult):
        """Update performance metrics."""
        self.performance_metrics["total_actions"] += 1
        if result.success:
            self.performance_metrics["successful_actions"] += 1
        
        self.performance_metrics["total_reward"] += result.reward
        self.performance_metrics["avg_reward"] = (
            self.performance_metrics["total_reward"] / 
            max(1, self.performance_metrics["total_actions"])
        )
        
        # Update history
        self.action_history.append(result.action)
        self.reward_history.append(result.reward)
        
        # Keep history bounded
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-50:]
            self.reward_history = self.reward_history[-50:]
    
    async def _update_memory(self, state: Dict[str, Any], 
                           action_plan: Dict[str, Any], 
                           result: ActionResult):
        """Update agent memory with insights."""
        try:
            # Generate insight from experience
            insight_content = f"Action: {action_plan.get('type', 'unknown')} | "
            insight_content += f"Success: {result.success} | Reward: {result.reward:.3f}"
            
            if result.gpt_reasoning:
                insight_content += f" | GPT: {result.gpt_reasoning[:100]}"
            
            # Determine insight type based on action result
            if result.success and result.reward > 0.5:
                insight_type = "tactical"
            elif not result.success:
                insight_type = "environmental"
            else:
                insight_type = "coordination"
            
            # Create insight
            insight = MemoryInsight(
                agent_id=self.agent_id,
                timestamp=time.time(),
                insight_type=insight_type,
                content=insight_content,
                confidence=min(1.0, abs(result.reward)),
                context={
                    "state_summary": str(state)[:200],
                    "action_type": action_plan.get('type', 'unknown'),
                    "success": result.success
                },
                relevance_tags=[]
            )
            
            # Store locally
            self.recent_insights.append(insight)
            if len(self.recent_insights) > 50:
                self.recent_insights = self.recent_insights[-30:]
            
            # Add to memory sync if available
            if self.memory_sync:
                self.memory_sync.add_insight(
                    agent_id=self.agent_id,
                    insight_type=insight_type,
                    content=insight_content,
                    context=insight.context,
                    confidence=insight.confidence
                )
                
        except Exception as e:
            logger.warning(f"Memory update failed for {self.agent_id}: {e}")
    
    async def _sync_with_agents(self):
        """Synchronize knowledge with other agents."""
        if not self.memory_sync:
            return
        
        try:
            # Get relevant insights from other agents
            relevant_insights = self.memory_sync.get_relevant_insights(
                agent_id=self.agent_id,
                query=self.current_context,
                max_results=5
            )
            
            if relevant_insights:
                # Update shared knowledge
                for insight in relevant_insights:
                    key = f"{insight.agent_id}_{insight.insight_type}"
                    self.shared_knowledge[key] = {
                        "content": insight.content,
                        "confidence": insight.confidence,
                        "timestamp": insight.timestamp
                    }
                
                # Keep shared knowledge bounded
                if len(self.shared_knowledge) > 20:
                    # Remove oldest entries
                    sorted_items = sorted(
                        self.shared_knowledge.items(),
                        key=lambda x: x[1]["timestamp"],
                        reverse=True
                    )
                    self.shared_knowledge = dict(sorted_items[:15])
                
                logger.debug(f"📡 {self.agent_id} synced {len(relevant_insights)} insights")
                
        except Exception as e:
            logger.warning(f"Agent sync failed for {self.agent_id}: {e}")
    
    def get_recent_insights(self) -> List[Dict[str, Any]]:
        """Get recent insights for memory synchronization."""
        return [insight.to_dict() for insight in self.recent_insights[-10:]]
    
    def receive_shared_insights(self, insights: List[Dict[str, Any]]):
        """Receive shared insights from other agents."""
        try:
            for insight_data in insights:
                # Process shared insight
                agent_id = insight_data.get("agent_id", "unknown")
                content = insight_data.get("content", "")
                confidence = insight_data.get("confidence", 0.5)
                timestamp = insight_data.get("timestamp", time.time())
                
                # Store in shared knowledge
                key = f"shared_{agent_id}_{len(self.shared_knowledge)}"
                self.shared_knowledge[key] = {
                    "content": content,
                    "confidence": confidence,
                    "timestamp": timestamp,
                    "source": agent_id
                }
                
            logger.debug(f"📥 {self.agent_id} received {len(insights)} shared insights")
            
        except Exception as e:
            logger.warning(f"Failed to receive shared insights for {self.agent_id}: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = dict(self.performance_metrics)
        summary.update({
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "current_context": self.current_context,
            "insights_count": len(self.recent_insights),
            "shared_knowledge_count": len(self.shared_knowledge),
            "training_active": self.trainer is not None,
            "device": str(self.device)
        })
        
        if self.trainer:
            summary["training_stats"] = {"active": True, "placeholder": "TODO"}
        
        return summary
    
    async def save_checkpoint(self, filepath: str):
        """Save agent state and networks."""
        try:
            checkpoint = {
                "config": asdict(self.config),
                "performance_metrics": self.performance_metrics,
                "policy_network_state": self.policy_network.state_dict(),
                "value_network_state": self.value_network.state_dict(),
                "recent_insights": [insight.to_dict() for insight in self.recent_insights],
                "shared_knowledge": self.shared_knowledge
            }
            
            if self.trainer:
                checkpoint["trainer_state"] = {"placeholder": "TODO"}
            
            torch.save(checkpoint, filepath)
            logger.info(f"💾 {self.agent_id} checkpoint saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {self.agent_id}: {e}")
    
    async def load_checkpoint(self, filepath: str):
        """Load agent state and networks."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load network states
            self.policy_network.load_state_dict(checkpoint["policy_network_state"])
            self.value_network.load_state_dict(checkpoint["value_network_state"])
            
            # Load metrics and memory
            self.performance_metrics = checkpoint.get("performance_metrics", self.performance_metrics)
            self.shared_knowledge = checkpoint.get("shared_knowledge", {})
            
            # Load insights
            insight_dicts = checkpoint.get("recent_insights", [])
            self.recent_insights = [MemoryInsight.from_dict(data) for data in insight_dicts]
            
            # Load trainer state (placeholder)
            if self.trainer and "trainer_state" in checkpoint:
                # TODO: Implement trainer state loading
                pass
            
            logger.info(f"📁 {self.agent_id} checkpoint loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {self.agent_id}: {e}")

def create_enhanced_agent_config(agent_id: str, agent_type: str, **kwargs) -> AgentConfig:
    """Create an optimized agent configuration."""
    return AgentConfig(
        agent_id=agent_id,
        agent_type=agent_type,
        state_dim=kwargs.get("state_dim", 256),
        action_dim=kwargs.get("action_dim", 64),
        hidden_dims=kwargs.get("hidden_dims", [512, 512, 256]),
        learning_rate=kwargs.get("learning_rate", 3e-4),
        batch_size=kwargs.get("batch_size", 64),
        memory_size=kwargs.get("memory_size", 10000),
        gpt_guidance=kwargs.get("gpt_guidance", True),
        use_enhanced_training=kwargs.get("use_enhanced_training", True),
        sync_interval=kwargs.get("sync_interval", 5.0)
    )
