#!/usr/bin/env python3
"""
Enhanced Multi-Agent Coordination System for ARIASKA_RL
🤝 Strategic Coordination | 🎯 Task Allocation | 📊 Performance Optimization | 🧠 Collective Intelligence
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict, deque
import asyncio
import threading
from queue import Queue, PriorityQueue

class TaskPriority(Enum):
    """Task priority levels for coordination."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class AgentRole(Enum):
    """Agent role definitions for coordination."""
    COMMANDER = "commander"  # Orion Agent
    ATTACKER = "attacker"    # Red Agent
    DEFENDER = "defender"    # Blue Agent
    SCOUT = "scout"          # Scout Agent
    STEALTH = "stealth"      # Shadow Agent

@dataclass
class CoordinationTask:
    """Task structure for multi-agent coordination."""
    task_id: str
    task_type: str
    priority: TaskPriority
    assigned_agents: List[str]
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

@dataclass
class AgentCapability:
    """Agent capability assessment for task allocation."""
    agent_id: str
    role: AgentRole
    skills: Dict[str, float]  # skill_name -> proficiency (0-1)
    current_load: float = 0.0
    max_capacity: float = 1.0
    availability: bool = True
    specializations: List[str] = field(default_factory=list)

class CoordinationMatrix:
    """Dynamic coordination matrix for agent interaction scoring."""
    
    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids
        self.matrix = np.eye(len(agent_ids))  # Initialize with identity
        self.interaction_history = defaultdict(list)
        self.success_history = defaultdict(list)
        
    def update_coordination_score(self, agent1: str, agent2: str, 
                                success: bool, task_complexity: float = 1.0):
        """Update coordination score based on task outcomes."""
        idx1 = self.agent_ids.index(agent1)
        idx2 = self.agent_ids.index(agent2)
        
        # Record interaction
        self.interaction_history[(agent1, agent2)].append({
            'success': success,
            'complexity': task_complexity,
            'timestamp': time.time()
        })
        
        # Calculate new coordination score
        recent_interactions = self.interaction_history[(agent1, agent2)][-10:]
        if recent_interactions:
            success_rate = sum(1 for i in recent_interactions if i['success']) / len(recent_interactions)
            avg_complexity = np.mean([i['complexity'] for i in recent_interactions])
            
            # Score considers both success rate and task complexity
            score = success_rate * (1 + 0.5 * avg_complexity)
            
            # Update matrix (symmetric)
            self.matrix[idx1][idx2] = score
            self.matrix[idx2][idx1] = score
    
    def get_best_partners(self, agent_id: str, exclude: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """Get best coordination partners for an agent."""
        if exclude is None:
            exclude = set()
            
        agent_idx = self.agent_ids.index(agent_id)
        scores = []
        
        for i, other_agent in enumerate(self.agent_ids):
            if other_agent != agent_id and other_agent not in exclude:
                score = self.matrix[agent_idx][i]
                scores.append((other_agent, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def get_team_synergy_score(self, team: List[str]) -> float:
        """Calculate overall team synergy score."""
        if len(team) < 2:
            return 1.0
            
        total_score = 0.0
        pair_count = 0
        
        for i, agent1 in enumerate(team):
            for j, agent2 in enumerate(team[i+1:], i+1):
                idx1 = self.agent_ids.index(agent1)
                idx2 = self.agent_ids.index(agent2)
                total_score += self.matrix[idx1][idx2]
                pair_count += 1
        
        return total_score / pair_count if pair_count > 0 else 1.0

class AdvancedTaskAllocator:
    """Advanced task allocation system using optimization algorithms."""
    
    def __init__(self, agents: Dict[str, AgentCapability]):
        self.agents = agents
        self.task_queue = PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = []
        self.coordination_matrix = CoordinationMatrix(list(agents.keys()))
        
        # Performance tracking
        self.allocation_history = []
        self.efficiency_metrics = defaultdict(list)
        
    def add_task(self, task: CoordinationTask):
        """Add task to allocation queue."""
        priority_value = task.priority.value
        self.task_queue.put((priority_value, task.created_at, task))
    
    def assess_agent_suitability(self, agent: AgentCapability, task: CoordinationTask) -> float:
        """Assess how suitable an agent is for a specific task."""
        if not agent.availability:
            return 0.0
            
        # Base suitability from skills
        required_skills = task.parameters.get('required_skills', {})
        skill_match = 0.0
        
        if required_skills:
            skill_scores = []
            for skill, required_level in required_skills.items():
                agent_level = agent.skills.get(skill, 0.0)
                skill_scores.append(min(agent_level / required_level, 1.0))
            skill_match = np.mean(skill_scores)
        else:
            skill_match = 0.7  # Default if no specific skills required
        
        # Role compatibility
        role_bonus = 0.0
        task_type = task.task_type
        
        role_compatibility = {
            AgentRole.COMMANDER: ['coordinate', 'oversee', 'plan'],
            AgentRole.ATTACKER: ['exploit', 'attack', 'penetrate'],
            AgentRole.DEFENDER: ['monitor', 'defend', 'analyze'],
            AgentRole.SCOUT: ['reconnaissance', 'scan', 'discover'],
            AgentRole.STEALTH: ['infiltrate', 'covert', 'stealth']
        }
        
        if any(keyword in task_type.lower() for keyword in role_compatibility[agent.role]):
            role_bonus = 0.3
        
        # Workload penalty
        workload_penalty = agent.current_load / agent.max_capacity
        
        # Specialization bonus
        spec_bonus = 0.2 if any(spec in task_type for spec in agent.specializations) else 0.0
        
        # Final suitability score
        suitability = (skill_match + role_bonus + spec_bonus) * (1 - workload_penalty)
        return max(0.0, min(1.0, float(suitability)))
    
    def allocate_optimal_team(self, task: CoordinationTask) -> List[str]:
        """Allocate optimal team for a task using advanced algorithms."""
        max_team_size = task.parameters.get('max_team_size', 3)
        min_team_size = task.parameters.get('min_team_size', 1)
        
        # Get suitability scores for all available agents
        agent_scores = []
        for agent_id, agent in self.agents.items():
            if agent.availability:
                score = self.assess_agent_suitability(agent, task)
                agent_scores.append((agent_id, score))
        
        # Sort by suitability
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy selection with coordination consideration
        selected_team = []
        
        for agent_id, score in agent_scores:
            if len(selected_team) >= max_team_size:
                break
                
            if score < 0.3:  # Minimum threshold
                continue
                
            # Check coordination with existing team members
            if selected_team:
                avg_coordination = np.mean([
                    self.coordination_matrix.matrix[
                        self.coordination_matrix.agent_ids.index(agent_id)][
                        self.coordination_matrix.agent_ids.index(team_member)]
                    for team_member in selected_team
                ])
                
                # Only add if coordination is reasonable
                if avg_coordination < 0.4 and len(selected_team) >= min_team_size:
                    continue
            
            selected_team.append(agent_id)
            
            if len(selected_team) >= min_team_size:
                # Check if current team is sufficient
                team_synergy = self.coordination_matrix.get_team_synergy_score(selected_team)
                if team_synergy > 0.7:
                    break
        
        return selected_team
    
    def process_task_queue(self) -> List[CoordinationTask]:
        """Process pending tasks and create allocations."""
        allocated_tasks = []
        
        while not self.task_queue.empty():
            priority, timestamp, task = self.task_queue.get()
            
            # Check dependencies
            if self._check_dependencies(task):
                team = self.allocate_optimal_team(task)
                
                if team:
                    task.assigned_agents = team
                    task.status = "allocated"
                    
                    # Update agent workloads
                    workload_per_agent = 1.0 / len(team)
                    for agent_id in team:
                        self.agents[agent_id].current_load += workload_per_agent
                        self.agents[agent_id].availability = self.agents[agent_id].current_load < self.agents[agent_id].max_capacity
                    
                    allocated_tasks.append(task)
                    self.active_tasks[task.task_id] = task
                    
                    # Record allocation
                    self.allocation_history.append({
                        'task_id': task.task_id,
                        'team': team,
                        'timestamp': time.time(),
                        'priority': task.priority.name
                    })
                else:
                    # Re-queue if no suitable team found
                    self.task_queue.put((priority, timestamp, task))
                    break
            else:
                # Re-queue if dependencies not met
                self.task_queue.put((priority, timestamp, task))
        
        return allocated_tasks
    
    def _check_dependencies(self, task: CoordinationTask) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
            
        for dep_task_id in task.dependencies:
            if dep_task_id in self.active_tasks:
                return False  # Dependency still active
            if not any(t.task_id == dep_task_id and t.status == "completed" 
                      for t in self.completed_tasks):
                return False  # Dependency not completed
        
        return True
    
    def complete_task(self, task_id: str, success: bool, result: Optional[Dict] = None):
        """Mark task as completed and update metrics."""
        if task_id not in self.active_tasks:
            return
            
        task = self.active_tasks[task_id]
        task.status = "completed" if success else "failed"
        task.completed_at = time.time()
        task.result = result or {}
        
        # Update coordination matrix
        team = task.assigned_agents
        task_complexity = task.parameters.get('complexity', 1.0)
        
        # Update pairwise coordination scores
        for i, agent1 in enumerate(team):
            for agent2 in team[i+1:]:
                self.coordination_matrix.update_coordination_score(
                    agent1, agent2, success, task_complexity
                )
        
        # Free up agent capacity
        workload_per_agent = 1.0 / len(team)
        for agent_id in team:
            self.agents[agent_id].current_load = max(0.0, 
                self.agents[agent_id].current_load - workload_per_agent)
            self.agents[agent_id].availability = True
        
        # Move to completed tasks
        self.completed_tasks.append(task)
        del self.active_tasks[task_id]
        
        # Update efficiency metrics
        duration = task.completed_at - task.created_at
        self.efficiency_metrics['task_duration'].append(duration)
        self.efficiency_metrics['success_rate'].append(success)
        self.efficiency_metrics['team_size'].append(len(team))

class StrategicCoordinator:
    """High-level strategic coordinator for multi-agent operations."""
    
    def __init__(self, agents: Dict[str, Any], task_allocator: AdvancedTaskAllocator):
        self.agents = agents
        self.task_allocator = task_allocator
        self.strategic_state = {
            'current_phase': 'reconnaissance',
            'objectives': [],
            'threats': [],
            'opportunities': []
        }
        
        # Communication channels
        self.command_queue = Queue()
        self.status_updates = defaultdict(list)
        
        # Strategic patterns
        self.learned_strategies = {}
        self.strategy_performance = defaultdict(list)
        
    def analyze_tactical_situation(self, environment_state: Dict) -> Dict[str, Any]:
        """Analyze current tactical situation and recommend strategies."""
        
        analysis = {
            'threat_level': self._assess_threat_level(environment_state),
            'opportunity_score': self._assess_opportunities(environment_state),
            'resource_allocation': self._recommend_resource_allocation(),
            'coordination_effectiveness': self._assess_coordination_effectiveness(),
            'recommended_phase': self._recommend_phase_transition()
        }
        
        return analysis
    
    def _assess_threat_level(self, env_state: Dict) -> float:
        """Assess current threat level based on environment state."""
        # Analyze defensive systems, monitoring, etc.
        defensive_strength = env_state.get('defensive_strength', 0.5)
        monitoring_level = env_state.get('monitoring_level', 0.5)
        active_countermeasures = env_state.get('active_countermeasures', 0)
        
        threat_level = (defensive_strength + monitoring_level + active_countermeasures * 0.3) / 2.3
        return min(1.0, threat_level)
    
    def _assess_opportunities(self, env_state: Dict) -> float:
        """Assess available opportunities in the environment."""
        vulnerabilities = len(env_state.get('vulnerabilities', []))
        open_ports = len(env_state.get('open_ports', []))
        weak_services = len(env_state.get('weak_services', []))
        
        # Normalize opportunity score
        max_opportunities = env_state.get('max_possible_opportunities', 10)
        opportunity_score = (vulnerabilities + open_ports + weak_services) / max_opportunities
        
        return min(1.0, opportunity_score)
    
    def _recommend_resource_allocation(self) -> Dict[str, float]:
        """Recommend resource allocation across agents."""
        allocation = {}
        
        # Base allocation by role
        total_capacity = sum(agent.max_capacity for agent in self.task_allocator.agents.values())
        
        for agent_id, agent in self.task_allocator.agents.items():
            base_allocation = agent.max_capacity / total_capacity
            
            # Adjust based on performance
            recent_performance = self._get_agent_performance(agent_id)
            performance_multiplier = 0.5 + recent_performance  # 0.5 to 1.5 range
            
            allocation[agent_id] = base_allocation * performance_multiplier
        
        # Normalize to sum to 1.0
        total_allocation = sum(allocation.values())
        for agent_id in allocation:
            allocation[agent_id] /= total_allocation
        
        return allocation
    
    def _assess_coordination_effectiveness(self) -> float:
        """Assess overall coordination effectiveness."""
        if not self.task_allocator.completed_tasks:
            return 0.5  # Default for no history
        
        recent_tasks = self.task_allocator.completed_tasks[-20:]  # Last 20 tasks
        
        # Success rate
        success_rate = sum(1 for task in recent_tasks if task.status == "completed") / len(recent_tasks)
        
        # Average team synergy
        avg_synergy = np.mean([
            self.task_allocator.coordination_matrix.get_team_synergy_score(task.assigned_agents)
            for task in recent_tasks
        ])
        
        # Task completion efficiency
        avg_duration = np.mean([
            task.completed_at - task.created_at for task in recent_tasks
            if task.completed_at is not None
        ])
        expected_duration = 30.0  # Expected 30 seconds per task
        efficiency = min(1.0, float(expected_duration / avg_duration))
        
        # Combined effectiveness score
        effectiveness = (success_rate * 0.5 + avg_synergy * 0.3 + efficiency * 0.2)
        return float(effectiveness)
    
    def _recommend_phase_transition(self) -> str:
        """Recommend next operational phase."""
        current_phase = self.strategic_state['current_phase']
        
        # Phase transition logic
        phase_transitions = {
            'reconnaissance': {
                'next': 'exploitation',
                'condition': lambda: self._get_reconnaissance_completion() > 0.7
            },
            'exploitation': {
                'next': 'persistence',
                'condition': lambda: self._get_exploitation_success() > 0.5
            },
            'persistence': {
                'next': 'exfiltration',
                'condition': lambda: self._get_persistence_established() > 0.6
            },
            'exfiltration': {
                'next': 'cleanup',
                'condition': lambda: self._get_data_extracted() > 0.8
            },
            'cleanup': {
                'next': 'reconnaissance',
                'condition': lambda: True  # Always ready for new cycle
            }
        }
        
        transition_info = phase_transitions.get(current_phase)
        if transition_info and transition_info['condition']():
            return transition_info['next']
        
        return current_phase
    
    def _get_agent_performance(self, agent_id: str) -> float:
        """Get recent performance score for an agent."""
        # Mock implementation - replace with actual performance tracking
        agent_tasks = [task for task in self.task_allocator.completed_tasks 
                      if agent_id in task.assigned_agents]
        
        if not agent_tasks:
            return 0.5
        
        recent_tasks = agent_tasks[-10:]  # Last 10 tasks
        success_rate = sum(1 for task in recent_tasks if task.status == "completed") / len(recent_tasks)
        
        return success_rate
    
    def _get_reconnaissance_completion(self) -> float:
        """Get reconnaissance phase completion percentage."""
        # Mock implementation
        return 0.8
    
    def _get_exploitation_success(self) -> float:
        """Get exploitation success rate."""
        # Mock implementation
        return 0.6
    
    def _get_persistence_established(self) -> float:
        """Get persistence establishment success."""
        # Mock implementation
        return 0.7
    
    def _get_data_extracted(self) -> float:
        """Get data exfiltration completion."""
        # Mock implementation
        return 0.9
    
    def generate_coordination_report(self) -> Dict[str, Any]:
        """Generate comprehensive coordination report."""
        
        # Agent performance summary
        agent_performance = {}
        for agent_id in self.agents.keys():
            agent_performance[agent_id] = {
                'performance_score': self._get_agent_performance(agent_id),
                'current_load': self.task_allocator.agents[agent_id].current_load,
                'specializations': self.task_allocator.agents[agent_id].specializations,
                'availability': self.task_allocator.agents[agent_id].availability
            }
        
        # Coordination matrix summary
        coord_matrix = self.task_allocator.coordination_matrix.matrix.tolist()
        
        # Task statistics
        task_stats = {
            'active_tasks': len(self.task_allocator.active_tasks),
            'completed_tasks': len(self.task_allocator.completed_tasks),
            'average_task_duration': np.mean(self.task_allocator.efficiency_metrics['task_duration']) if self.task_allocator.efficiency_metrics['task_duration'] else 0,
            'success_rate': np.mean(self.task_allocator.efficiency_metrics['success_rate']) if self.task_allocator.efficiency_metrics['success_rate'] else 0
        }
        
        return {
            'timestamp': time.time(),
            'strategic_state': self.strategic_state,
            'agent_performance': agent_performance,
            'coordination_matrix': coord_matrix,
            'task_statistics': task_stats,
            'coordination_effectiveness': self._assess_coordination_effectiveness(),
            'recommendations': self._generate_strategic_recommendations()
        }
    
    def _generate_strategic_recommendations(self) -> List[str]:
        """Generate strategic recommendations based on current state."""
        recommendations = []
        
        # Check coordination effectiveness
        effectiveness = self._assess_coordination_effectiveness()
        if effectiveness < 0.6:
            recommendations.append("Improve agent coordination through better task allocation")
        
        # Check individual agent performance
        for agent_id in self.agents.keys():
            performance = self._get_agent_performance(agent_id)
            if performance < 0.4:
                recommendations.append(f"Focus on improving {agent_id} performance through additional training")
        
        # Check resource utilization
        avg_load = np.mean([agent.current_load for agent in self.task_allocator.agents.values()])
        if avg_load < 0.3:
            recommendations.append("Increase task complexity or frequency to better utilize agent capacity")
        elif avg_load > 0.8:
            recommendations.append("Consider load balancing or task prioritization improvements")
        
        return recommendations
