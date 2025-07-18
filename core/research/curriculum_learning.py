"""
Advanced Curriculum Learning Framework for ARIASKA_RL

Implements adaptive curriculum learning with difficulty progression,
performance-based adaptation, and multi-agent coordination.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from rich.console import Console

console = Console()

class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class CurriculumStage:
    """Single stage in curriculum progression"""
    stage_id: str
    difficulty: DifficultyLevel
    learning_objectives: List[str]
    environment_params: Dict[str, Any]
    success_criteria: Dict[str, float]
    prerequisites: List[str]
    estimated_episodes: int
    description: str

@dataclass
class LearningProgress:
    """Track learning progress for curriculum adaptation"""
    agent_id: str
    current_stage: str
    stages_completed: List[str]
    performance_history: Dict[str, List[float]]
    skill_assessments: Dict[str, float]
    adaptation_history: List[Dict[str, Any]]
    total_episodes: int
    last_updated: str

class CurriculumLearning:
    """
    Advanced curriculum learning system for ARIASKA_RL.
    
    Features:
    - Adaptive difficulty progression based on performance
    - Multi-agent coordination and peer learning
    - Skill-based curriculum branching
    - Performance prediction and optimization
    - Knowledge transfer between stages
    """
    
    def __init__(self, curriculum_dir: str = "curriculum"):
        self.curriculum_dir = curriculum_dir
        self.stages_dir = os.path.join(curriculum_dir, "stages")
        self.progress_dir = os.path.join(curriculum_dir, "progress")
        
        os.makedirs(self.stages_dir, exist_ok=True)
        os.makedirs(self.progress_dir, exist_ok=True)
        
        self.curriculum_stages: Dict[str, CurriculumStage] = {}
        self.agent_progress: Dict[str, LearningProgress] = {}
        
        self._initialize_default_curriculum()
        console.print(f"[green]✓[/green] CurriculumLearning initialized")
    
    def _initialize_default_curriculum(self):
        """Create default cybersecurity curriculum stages"""
        
        # Stage 1: Basic Reconnaissance
        stage1 = CurriculumStage(
            stage_id="basic_recon",
            difficulty=DifficultyLevel.BEGINNER,
            learning_objectives=[
                "Learn to perform network discovery",
                "Identify open ports and services",
                "Understand basic scanning techniques"
            ],
            environment_params={
                "target_complexity": "simple",
                "network_size": "small",
                "defensive_measures": "minimal",
                "time_pressure": "none"
            },
            success_criteria={
                "avg_reward": 10.0,
                "success_rate": 0.7,
                "phase_accuracy": 0.8
            },
            prerequisites=[],
            estimated_episodes=20,
            description="Introduction to network reconnaissance and discovery"
        )
        
        # Stage 2: Advanced Scanning
        stage2 = CurriculumStage(
            stage_id="advanced_scanning",
            difficulty=DifficultyLevel.INTERMEDIATE,
            learning_objectives=[
                "Master advanced scanning techniques",
                "Learn to evade basic detection",
                "Understand service enumeration"
            ],
            environment_params={
                "target_complexity": "moderate",
                "network_size": "medium",
                "defensive_measures": "basic",
                "time_pressure": "low"
            },
            success_criteria={
                "avg_reward": 15.0,
                "success_rate": 0.6,
                "phase_accuracy": 0.75
            },
            prerequisites=["basic_recon"],
            estimated_episodes=30,
            description="Advanced scanning and enumeration techniques"
        )
        
        # Stage 3: Vulnerability Assessment
        stage3 = CurriculumStage(
            stage_id="vuln_assessment",
            difficulty=DifficultyLevel.INTERMEDIATE,
            learning_objectives=[
                "Identify common vulnerabilities",
                "Assess exploitability",
                "Prioritize attack vectors"
            ],
            environment_params={
                "target_complexity": "moderate",
                "network_size": "medium",
                "defensive_measures": "moderate",
                "time_pressure": "moderate"
            },
            success_criteria={
                "avg_reward": 20.0,
                "success_rate": 0.5,
                "phase_accuracy": 0.7
            },
            prerequisites=["advanced_scanning"],
            estimated_episodes=40,
            description="Comprehensive vulnerability assessment and analysis"
        )
        
        # Stage 4: Exploitation
        stage4 = CurriculumStage(
            stage_id="exploitation",
            difficulty=DifficultyLevel.ADVANCED,
            learning_objectives=[
                "Execute successful exploits",
                "Maintain persistence",
                "Escalate privileges"
            ],
            environment_params={
                "target_complexity": "complex",
                "network_size": "large",
                "defensive_measures": "advanced",
                "time_pressure": "high"
            },
            success_criteria={
                "avg_reward": 25.0,
                "success_rate": 0.4,
                "phase_accuracy": 0.65
            },
            prerequisites=["vuln_assessment"],
            estimated_episodes=50,
            description="Advanced exploitation and privilege escalation"
        )
        
        # Stage 5: Post-Exploitation
        stage5 = CurriculumStage(
            stage_id="post_exploitation",
            difficulty=DifficultyLevel.EXPERT,
            learning_objectives=[
                "Maintain stealth and persistence",
                "Lateral movement techniques",
                "Data exfiltration methods"
            ],
            environment_params={
                "target_complexity": "expert",
                "network_size": "enterprise",
                "defensive_measures": "military_grade",
                "time_pressure": "extreme"
            },
            success_criteria={
                "avg_reward": 30.0,
                "success_rate": 0.3,
                "phase_accuracy": 0.6
            },
            prerequisites=["exploitation"],
            estimated_episodes=60,
            description="Advanced post-exploitation and stealth techniques"
        )
        
        # Register stages
        for stage in [stage1, stage2, stage3, stage4, stage5]:
            self.curriculum_stages[stage.stage_id] = stage
            
            # Save stage configuration
            stage_path = os.path.join(self.stages_dir, f"{stage.stage_id}.json")
            with open(stage_path, 'w') as f:
                json.dump(asdict(stage), f, indent=2, default=str)
    
    def initialize_agent_progress(self, agent_id: str):
        """Initialize learning progress tracking for an agent"""
        
        progress = LearningProgress(
            agent_id=agent_id,
            current_stage="basic_recon",
            stages_completed=[],
            performance_history={},
            skill_assessments={},
            adaptation_history=[],
            total_episodes=0,
            last_updated=datetime.now().isoformat()
        )
        
        self.agent_progress[agent_id] = progress
        self._save_agent_progress(agent_id)
        
        console.print(f"[cyan]📚[/cyan] Initialized curriculum progress for {agent_id}")
    
    def get_current_curriculum_config(self, agent_id: str) -> Dict[str, Any]:
        """Get current curriculum configuration for agent"""
        
        if agent_id not in self.agent_progress:
            self.initialize_agent_progress(agent_id)
        
        progress = self.agent_progress[agent_id]
        current_stage = self.curriculum_stages[progress.current_stage]
        
        return {
            "stage_id": current_stage.stage_id,
            "difficulty": current_stage.difficulty.value,
            "environment_params": current_stage.environment_params,
            "learning_objectives": current_stage.learning_objectives,
            "success_criteria": current_stage.success_criteria,
            "estimated_episodes": current_stage.estimated_episodes
        }
    
    def update_agent_performance(self, agent_id: str, episode: int, 
                               performance_metrics: Dict[str, float]):
        """Update agent performance and assess curriculum progression"""
        
        if agent_id not in self.agent_progress:
            self.initialize_agent_progress(agent_id)
        
        progress = self.agent_progress[agent_id]
        current_stage = self.curriculum_stages[progress.current_stage]
        
        # Update performance history
        for metric, value in performance_metrics.items():
            if metric not in progress.performance_history:
                progress.performance_history[metric] = []
            progress.performance_history[metric].append(value)
        
        progress.total_episodes += 1
        progress.last_updated = datetime.now().isoformat()
        
        # Assess if ready for next stage
        should_advance = self._assess_stage_completion(agent_id, current_stage)
        
        if should_advance:
            next_stage = self._get_next_stage(current_stage.stage_id)
            if next_stage:
                self._advance_to_next_stage(agent_id, next_stage)
        else:
            # Check if need to adapt current stage difficulty
            adaptation = self._assess_difficulty_adaptation(agent_id, current_stage)
            if adaptation:
                self._apply_difficulty_adaptation(agent_id, adaptation)
        
        self._save_agent_progress(agent_id)
    
    def _assess_stage_completion(self, agent_id: str, stage: CurriculumStage) -> bool:
        """Assess if agent has mastered current curriculum stage"""
        
        progress = self.agent_progress[agent_id]
        
        # Need minimum episodes before assessment
        if progress.total_episodes < stage.estimated_episodes * 0.5:
            return False
        
        # Check recent performance (last 10 episodes)
        recent_window = 10
        stage_mastery = True
        
        for criterion, threshold in stage.success_criteria.items():
            if criterion in progress.performance_history:
                recent_performance = progress.performance_history[criterion][-recent_window:]
                if recent_performance:
                    avg_performance = np.mean(recent_performance)
                    if avg_performance < threshold:
                        stage_mastery = False
                        break
        
        # Additional stability check - performance should be consistent
        if stage_mastery:
            for criterion in stage.success_criteria.keys():
                if criterion in progress.performance_history:
                    recent_performance = progress.performance_history[criterion][-recent_window:]
                    if recent_performance and len(recent_performance) > 5:
                        stability = 1.0 - (np.std(recent_performance) / (np.mean(recent_performance) + 1e-6))
                        if stability < 0.7:  # Require 70% stability
                            stage_mastery = False
                            break
        
        return stage_mastery
    
    def _get_next_stage(self, current_stage_id: str) -> Optional[str]:
        """Get next stage in curriculum progression"""
        
        # Find stages that have current stage as prerequisite
        for stage_id, stage in self.curriculum_stages.items():
            if current_stage_id in stage.prerequisites:
                return stage_id
        
        return None
    
    def _advance_to_next_stage(self, agent_id: str, next_stage_id: str):
        """Advance agent to next curriculum stage"""
        
        progress = self.agent_progress[agent_id]
        previous_stage = progress.current_stage
        
        # Mark current stage as completed
        progress.stages_completed.append(previous_stage)
        progress.current_stage = next_stage_id
        
        # Record advancement
        advancement = {
            "type": "stage_advancement",
            "from_stage": previous_stage,
            "to_stage": next_stage_id,
            "episode": progress.total_episodes,
            "timestamp": datetime.now().isoformat()
        }
        progress.adaptation_history.append(advancement)
        
        console.print(f"[green]🎓[/green] {agent_id} advanced to stage: {next_stage_id}")
    
    def _assess_difficulty_adaptation(self, agent_id: str, stage: CurriculumStage) -> Optional[Dict[str, Any]]:
        """Assess if stage difficulty should be adapted"""
        
        progress = self.agent_progress[agent_id]
        
        # Need sufficient data for assessment
        if progress.total_episodes < 10:
            return None
        
        # Analyze recent performance trend
        recent_rewards = progress.performance_history.get('avg_reward', [])[-10:]
        if not recent_rewards:
            return None
        
        # Calculate performance trend
        if len(recent_rewards) >= 5:
            recent_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
            avg_recent = np.mean(recent_rewards)
            target_reward = stage.success_criteria['avg_reward']
            
            # Too easy - increase difficulty
            if avg_recent > target_reward * 1.2 and recent_trend > 0:
                return {
                    "type": "increase_difficulty",
                    "reason": "performance_too_high",
                    "adjustments": {
                        "defensive_measures": "enhanced",
                        "time_pressure": "increased",
                        "target_complexity": "higher"
                    }
                }
            
            # Too hard - decrease difficulty  
            elif avg_recent < target_reward * 0.5 and recent_trend < 0:
                return {
                    "type": "decrease_difficulty", 
                    "reason": "performance_too_low",
                    "adjustments": {
                        "defensive_measures": "reduced",
                        "time_pressure": "relaxed",
                        "target_complexity": "simplified"
                    }
                }
        
        return None
    
    def _apply_difficulty_adaptation(self, agent_id: str, adaptation: Dict[str, Any]):
        """Apply difficulty adaptation to current stage"""
        
        progress = self.agent_progress[agent_id]
        current_stage = self.curriculum_stages[progress.current_stage]
        
        # Apply parameter adjustments
        for param, adjustment in adaptation["adjustments"].items():
            if param in current_stage.environment_params:
                current_stage.environment_params[param] = adjustment
        
        # Record adaptation
        adaptation["episode"] = progress.total_episodes
        adaptation["timestamp"] = datetime.now().isoformat()
        progress.adaptation_history.append(adaptation)
        
        console.print(f"[yellow]⚙️[/yellow] Adapted difficulty for {agent_id}: {adaptation['type']}")
    
    def get_peer_learning_insights(self, agent_id: str) -> Dict[str, Any]:
        """Get insights from peer agents' learning progress"""
        
        if agent_id not in self.agent_progress:
            return {}
        
        current_stage = self.agent_progress[agent_id].current_stage
        insights = {
            "peer_performance": {},
            "successful_strategies": [],
            "common_challenges": [],
            "recommended_focus": []
        }
        
        # Analyze peer performance in same stage
        peer_performances = []
        for other_agent, other_progress in self.agent_progress.items():
            if other_agent != agent_id and other_progress.current_stage == current_stage:
                if 'avg_reward' in other_progress.performance_history:
                    recent_performance = other_progress.performance_history['avg_reward'][-10:]
                    if recent_performance:
                        peer_performances.append({
                            "agent": other_agent,
                            "avg_reward": np.mean(recent_performance),
                            "episodes": len(other_progress.performance_history.get('avg_reward', []))
                        })
        
        if peer_performances:
            # Find top performing peers
            top_peers = sorted(peer_performances, key=lambda x: x['avg_reward'], reverse=True)[:3]
            insights["peer_performance"] = {
                "top_performers": top_peers,
                "peer_average": np.mean([p['avg_reward'] for p in peer_performances]),
                "your_ranking": len([p for p in peer_performances if p['avg_reward'] > 
                                  np.mean(self.agent_progress[agent_id].performance_history.get('avg_reward', [])[-10:] or [0])]) + 1
            }
        
        return insights
    
    def generate_curriculum_report(self, agent_id: str) -> str:
        """Generate comprehensive curriculum progress report"""
        
        if agent_id not in self.agent_progress:
            return "No curriculum progress found for agent"
        
        progress = self.agent_progress[agent_id]
        current_stage = self.curriculum_stages[progress.current_stage]
        
        report = f"""# Curriculum Progress Report: {agent_id}

## Current Status
- **Current Stage**: {current_stage.stage_id} ({current_stage.difficulty.value})
- **Total Episodes**: {progress.total_episodes}
- **Stages Completed**: {len(progress.stages_completed)}
- **Last Updated**: {progress.last_updated}

## Learning Objectives Progress
"""
        
        for objective in current_stage.learning_objectives:
            report += f"- {objective}\n"
        
        report += f"""
## Performance Metrics
"""
        
        for metric, values in progress.performance_history.items():
            if values:
                recent_avg = np.mean(values[-10:])
                overall_avg = np.mean(values)
                target = current_stage.success_criteria.get(metric, 0)
                
                report += f"- **{metric}**: Recent={recent_avg:.3f}, Overall={overall_avg:.3f}, Target={target:.3f}\n"
        
        report += f"""
## Adaptation History
"""
        
        for adaptation in progress.adaptation_history[-5:]:  # Last 5 adaptations
            report += f"- Episode {adaptation.get('episode', 'N/A')}: {adaptation.get('type', 'Unknown')} - {adaptation.get('reason', 'No reason')}\n"
        
        # Recommendations
        report += f"""
## Recommendations
"""
        
        recommendations = self._generate_curriculum_recommendations(agent_id)
        for rec in recommendations:
            report += f"- {rec}\n"
        
        return report
    
    def _generate_curriculum_recommendations(self, agent_id: str) -> List[str]:
        """Generate personalized curriculum recommendations"""
        
        progress = self.agent_progress[agent_id]
        current_stage = self.curriculum_stages[progress.current_stage]
        recommendations = []
        
        # Performance-based recommendations
        for metric, values in progress.performance_history.items():
            if values and len(values) >= 5:
                recent_trend = np.polyfit(range(len(values[-10:])), values[-10:], 1)[0]
                target = current_stage.success_criteria.get(metric, 0)
                current_avg = np.mean(values[-5:])
                
                if current_avg < target * 0.8:
                    recommendations.append(f"Focus on improving {metric} - currently {current_avg:.3f}, target {target:.3f}")
                
                if recent_trend < -0.1:
                    recommendations.append(f"Address declining {metric} performance")
        
        # Peer learning recommendations
        peer_insights = self.get_peer_learning_insights(agent_id)
        if peer_insights.get("peer_performance"):
            ranking = peer_insights["peer_performance"]["your_ranking"]
            total_peers = len(peer_insights["peer_performance"]["top_performers"])
            
            if ranking > total_peers * 0.7:
                recommendations.append("Consider reviewing top performer strategies")
        
        return recommendations
    
    def _save_agent_progress(self, agent_id: str):
        """Save agent progress to file"""
        
        progress_path = os.path.join(self.progress_dir, f"{agent_id}_progress.json")
        with open(progress_path, 'w') as f:
            json.dump(asdict(self.agent_progress[agent_id]), f, indent=2, default=str)
    
    def load_agent_progress(self, agent_id: str) -> bool:
        """Load saved agent progress"""
        
        progress_path = os.path.join(self.progress_dir, f"{agent_id}_progress.json")
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r') as f:
                    progress_data = json.load(f)
                    self.agent_progress[agent_id] = LearningProgress(**progress_data)
                return True
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Error loading progress for {agent_id}: {e}")
        
        return False
    
    def get_curriculum_analytics(self) -> Dict[str, Any]:
        """Get analytics across all agents in curriculum"""
        
        analytics = {
            "total_agents": len(self.agent_progress),
            "stage_distribution": {},
            "completion_rates": {},
            "average_progression_time": {},
            "common_bottlenecks": []
        }
        
        # Stage distribution
        for agent_id, progress in self.agent_progress.items():
            stage = progress.current_stage
            analytics["stage_distribution"][stage] = analytics["stage_distribution"].get(stage, 0) + 1
        
        # Completion rates by stage
        for stage_id in self.curriculum_stages.keys():
            completed = len([p for p in self.agent_progress.values() if stage_id in p.stages_completed])
            started = len([p for p in self.agent_progress.values() 
                         if stage_id in p.stages_completed or p.current_stage == stage_id])
            
            if started > 0:
                analytics["completion_rates"][stage_id] = completed / started
        
        return analytics