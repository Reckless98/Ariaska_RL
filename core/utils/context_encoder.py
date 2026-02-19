"""
context_encoder.py — Advanced Context Summarization and Encoding
Provides utilities for summarizing agent histories, encoding contexts for LLM consumption,
and maintaining compact representations of agent experiences for strategic planning.
"""

import time
import json
import re
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict

# Attempt to import potential GPT management classes
try:
    from core.gpt_manager import GPTManager
except ImportError:
    GPTManager = None

logger = logging.getLogger(__name__)

class ContextEncoder:
    """
    Provides methods for summarizing agent histories, encoding contexts,
    and maintaining compact representations of agent experiences.
    Features:
    - Token-efficient context summarization for LLMs
    - Phase and episode summary generation
    - Contextual compression for memory efficiency
    - Agent history digest for strategic planning
    """
    
    @staticmethod
    def optimize_for_llm_prompt(
        context: Dict[str, Any], 
        max_chars: int = 1000, 
        priority_keys: List[str] = None
    ) -> str:
        """
        Optimize a complex context dictionary for LLM prompts by prioritizing
        the most important keys and truncating to fit within token limits.
        
        Args:
            context: The context dictionary to optimize
            max_chars: Maximum character length for output string
            priority_keys: Optional list of keys to prioritize
            
        Returns:
            A compact string representation of the context
        """
        if not context:
            return ""
            
        # Default priority keys if not specified
        if not priority_keys:
            priority_keys = [
                "phase", "targets", "ports", "services", "vulnerabilities",
                "credentials", "exploited_hosts", "privilege_level"
            ]
        
        # Create a priority order for all keys
        all_keys = list(context.keys())
        priority_dict = {k: i for i, k in enumerate(priority_keys)}
        
        # Sort keys by priority (priority keys first, then alphabetically)
        sorted_keys = sorted(
            all_keys,
            key=lambda k: (priority_dict.get(k, len(priority_keys)), k)
        )
        
        # Build context string with priority-based truncation
        result = []
        chars_used = 0
        
        # First pass: Add high-priority items
        for key in sorted_keys:
            if key in priority_keys:
                value = context[key]
                
                # Format the value based on type
                if isinstance(value, dict):
                    # Compact dict representation
                    formatted = f"{key}: " + json.dumps(value, separators=(',', ':'))
                    
                    # Truncate if too long
                    if len(formatted) > 200:  # Guard against very large dicts
                        formatted = f"{key}: " + str(value)[:197] + "..."
                elif isinstance(value, list):
                    # Handle lists specially - limit number of items
                    if len(value) > 5:
                        items = ", ".join(str(v) for v in value[:5])
                        formatted = f"{key}: [{items}, ...+{len(value)-5} more]"
                    else:
                        formatted = f"{key}: {value}"
                else:
                    formatted = f"{key}: {value}"
                
                result.append(formatted)
                chars_used += len(formatted)
        
        # Second pass: Add other fields if space permits
        for key in sorted_keys:
            if key not in priority_keys and chars_used < max_chars:
                value = context[key]
                
                # Skip verbose fields completely
                if key in ["verbose_logs", "scan_history", "exploit_history", "discovered_data"]:
                    continue
                
                # Format based on type (similar to above)
                if isinstance(value, dict):
                    formatted = f"{key}: " + json.dumps(value, separators=(',', ':'))
                    if len(formatted) > 100:
                        formatted = f"{key}: " + str(value)[:97] + "..."
                elif isinstance(value, list):
                    if len(value) > 3:
                        items = ", ".join(str(v) for v in value[:3])
                        formatted = f"{key}: [{items}, ...+{len(value)-3} more]"
                    else:
                        formatted = f"{key}: {value}"
                else:
                    formatted = f"{key}: {value}"
                
                # Add only if fits within our character limit
                if chars_used + len(formatted) <= max_chars:
                    result.append(formatted)
                    chars_used += len(formatted)
        
        # Join with newlines for readability
        return "\n".join(result)
    
    @staticmethod
    def create_summary_prompt(
        agent_id: str,
        transitions: List[Dict[str, Any]],
        phase: str = None,
        episode_id: str = None
    ) -> str:
        """
        Create a prompt for LLMs to summarize a set of agent transitions.
        
        Args:
            agent_id: ID of the agent
            transitions: List of transitions to summarize
            phase: Optional phase name
            episode_id: Optional episode ID
            
        Returns:
            A prompt for LLM summarization
        """
        context = {
            "agent_id": agent_id,
            "phase": phase or "unknown",
            "episode_id": episode_id or "unknown",
            "num_transitions": len(transitions),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Extract key information
        rewards = [t.get("reward", 0) for t in transitions]
        actions = [t.get("action", "") for t in transitions]
        metadata_flags = defaultdict(int)
        
        for t in transitions:
            metadata = t.get("metadata", {})
            for flag, value in metadata.items():
                if isinstance(value, bool) and value:
                    metadata_flags[flag] += 1
        
        # Add statistics
        context["total_reward"] = sum(rewards)
        context["avg_reward"] = sum(rewards) / len(rewards) if rewards else 0
        context["significant_events"] = dict(metadata_flags)
        
        # Add compact action history
        action_preview = [str(a) for a in actions[:5]]
        if len(actions) > 5:
            action_preview.append(f"...+{len(actions)-5} more")
        context["actions"] = action_preview
        
        # Create the prompt
        prompt = f"""
Please create a concise summary of the following agent activity:

Agent: {context['agent_id']}
Phase: {context['phase']}
Episode: {context['episode_id']}
Time: {context['time']}

Number of actions: {context['num_transitions']}
Total reward: {context['total_reward']:.2f}
Average reward: {context['avg_reward']:.2f}

Significant events:
{json.dumps(context['significant_events'], indent=2)}

Sample actions:
{json.dumps(context['actions'], indent=2)}

Please provide:
1. A one-sentence summary of what the agent accomplished
2. Key discoveries or achievements (hosts found, services enumerated, exploits executed)
3. Challenges encountered (failed attempts, detections)
4. Most important action taken
5. Strategic recommendations for future actions

Format your response as a JSON object with these fields:
{{
    "summary": "One sentence summary",
    "key_discoveries": ["discovery1", "discovery2", ...],
    "challenges": ["challenge1", "challenge2", ...],
    "critical_action": "most important action",
    "recommendations": ["recommendation1", "recommendation2", ...]
}}
"""
        return prompt
    
    @staticmethod
    def summarize_transitions(
        agent_id: str,
        transitions: List[Dict[str, Any]],
        phase: str = None,
        episode_id: str = None,
        memory_router = None
    ) -> Dict[str, Any]:
        """
        Summarize a set of agent transitions, using GPT if available.
        Falls back to statistical summary if GPT unavailable.
        
        Args:
            agent_id: ID of the agent
            transitions: List of transitions to summarize
            phase: Optional phase name
            episode_id: Optional episode ID
            memory_router: Optional memory_router for storing the summary
            
        Returns:
            A summary dictionary
        """
        if not transitions:
            return {
                "summary": "No agent activity to summarize",
                "key_discoveries": [],
                "challenges": [],
                "critical_action": None,
                "recommendations": []
            }
        
        # Create the prompt for GPT
        prompt = ContextEncoder.create_summary_prompt(agent_id, transitions, phase, episode_id)
        
        # Try to use GPTManager if available
        if GPTManager:
            try:
                gpt = GPTManager.get_instance()
                response = gpt.gpt_request(
                    prompt=prompt,
                    system="You are an expert cybersecurity AI assistant that creates concise summaries of agent activities.",
                    model="gpt-5.2-codex",  # Phase 38: upgraded from codex-mini
                    max_tokens=500
                )
                
                # Parse JSON response
                try:
                    # Extract JSON if wrapped in backticks
                    json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = response
                        
                    summary = json.loads(json_str)
                    
                    # Store summary if memory router provided
                    if memory_router:
                        if phase:
                            memory_router.add_summary(agent_id, summary, phase=phase)
                        if episode_id:
                            memory_router.add_summary(agent_id, summary, episode_id=episode_id)
                    
                    return summary
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse GPT summary response as JSON, falling back to statistical summary")
                    # Fall back to statistical summary
            except Exception as e:
                logger.warning(f"Failed to get GPT summary: {e}, falling back to statistical summary")
                # Fall back to statistical summary
        
        # Statistical summary (fallback method)
        rewards = [t.get("reward", 0) for t in transitions]
        actions = [t.get("action", "") for t in transitions]
        metadata_counts = defaultdict(int)
        
        for t in transitions:
            metadata = t.get("metadata", {})
            for flag, value in metadata.items():
                if isinstance(value, bool) and value:
                    metadata_counts[flag] += 1
        
        # Find transition with highest reward as critical action
        max_reward_idx = rewards.index(max(rewards)) if rewards else -1
        critical_action = actions[max_reward_idx] if max_reward_idx >= 0 else None
        
        # Create statistical summary
        summary = {
            "summary": f"Agent {agent_id} performed {len(transitions)} actions in {phase or 'unknown'} phase with total reward {sum(rewards):.2f}",
            "key_discoveries": [f"{k}: {v}" for k, v in metadata_counts.items() if "discovery" in k.lower() or "found" in k.lower()],
            "challenges": [f"{k}: {v}" for k, v in metadata_counts.items() if "failed" in k.lower() or "detection" in k.lower()],
            "critical_action": critical_action,
            "recommendations": []
        }
        
        # Store summary if memory router provided
        if memory_router:
            if phase:
                memory_router.add_summary(agent_id, summary, phase=phase)
            if episode_id:
                memory_router.add_summary(agent_id, summary, episode_id=episode_id)
                
        return summary
    
    @staticmethod
    def digest_agent_history(
        agent_id: str, 
        memory_router, 
        max_items: int = 5
    ) -> Dict[str, Any]:
        """
        Create a comprehensive but compact digest of an agent's history.
        
        Args:
            agent_id: ID of the agent
            memory_router: Memory router to retrieve agent data
            max_items: Maximum number of items per category
            
        Returns:
            A digest dictionary
        """
        if not memory_router:
            return {"error": "No memory_router provided"}
            
        digest = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "recent_actions": [],
            "phase_summaries": {},
            "episode_summaries": [],
            "token_usage": memory_router.get_token_usage(agent_id),
            "statistics": {}
        }
        
        # Get recent transitions
        recent = memory_router.get_recent_transitions(agent_id, limit=max_items)
        digest["recent_actions"] = [
            {
                "action": t.action,
                "reward": t.reward,
                "phase": t.phase,
                "metadata": t.metadata
            } for t in recent
        ]
        
        # Get phase summaries (for recon, exploit, etc.)
        for phase in ["recon", "exploit", "privilege_escalation", "lateral_movement", "exfiltration"]:
            summary = memory_router.get_summary(agent_id, phase=phase)
            if summary:
                digest["phase_summaries"][phase] = summary
        
        # Get stats
        stats = memory_router.get_stats(agent_id)
        if agent_id in stats.get("agents", {}):
            digest["statistics"] = stats["agents"][agent_id]
        
        return digest
    
    @staticmethod
    def generate_strategy_context(
        agent_id: str,
        memory_router,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a strategy-oriented context for OrionAgent planning.
        
        Args:
            agent_id: ID of the agent (typically OrionAgent)
            memory_router: Memory router to retrieve agent data
            max_tokens: Approximate token limit for context
            
        Returns:
            A formatted context string optimized for strategic planning
        """
        if not memory_router:
            return "No memory_router provided"
            
        # Build agent digests
        agent_ids = ["RedAgent", "BlueAgent", "ScoutAgent", "ShadowAgent", "OrionAgent"]
        digests = {}
        
        for aid in agent_ids:
            digests[aid] = ContextEncoder.digest_agent_history(aid, memory_router)
        
        # Format context
        context = []
        context.append(f"=== STRATEGIC CONTEXT FOR {agent_id} ===\n")
        
        # Overall mission status
        all_phases = {}
        for aid, digest in digests.items():
            all_phases.update(digest.get("phase_summaries", {}))
            
        context.append("=== MISSION STATUS ===")
        for phase, summary in all_phases.items():
            context.append(f"{phase.upper()}: {summary.get('summary', 'No data')}")
        
        # Agent-specific summaries
        context.append("\n=== AGENT STATUS ===")
        for aid, digest in digests.items():
            token_usage = digest.get("token_usage", {}).get(aid, 0)
            recent = digest.get("recent_actions", [])
            recent_str = ", ".join([f"{a.get('action', 'unknown')} ({a.get('reward', 0):.1f})" for a in recent[:3]])
            
            context.append(f"{aid}: {token_usage} tokens used, recent: {recent_str}")
        
        # Phase-specific details (in order of typical attack flow)
        phase_order = ["recon", "exploit", "privilege_escalation", "lateral_movement", "exfiltration"]
        
        for phase in phase_order:
            if phase in all_phases:
                context.append(f"\n=== {phase.upper()} PHASE ===")
                summary = all_phases[phase]
                
                # Key discoveries
                discoveries = summary.get("key_discoveries", [])
                if discoveries:
                    context.append("Key Discoveries:")
                    for d in discoveries[:3]:  # Limit to 3 for token efficiency
                        context.append(f"- {d}")
                
                # Critical action
                if summary.get("critical_action"):
                    context.append(f"Critical Action: {summary.get('critical_action')}")
                
                # Recommendations (if any)
                recommendations = summary.get("recommendations", [])
                if recommendations:
                    context.append("Recommendations:")
                    for r in recommendations[:2]:  # Limit to 2 for token efficiency
                        context.append(f"- {r}")
        
        # Join all context parts
        result = "\n".join(context)
        
        # Truncate if needed (very rough approximation)
        max_chars = max_tokens * 4  # ~4 chars per token as rough estimate
        if len(result) > max_chars:
            result = result[:max_chars - 100] + "...[truncated for token efficiency]"
            
        return result


if __name__ == "__main__":
    # Example usage
    sample_context = {
        "phase": "exploit",
        "targets": ["10.10.10.10", "10.10.10.15"],
        "ports": {
            "10.10.10.10": [22, 80, 443],
            "10.10.10.15": [22, 21, 3389]
        },
        "services": {
            "10.10.10.10": {"22": "OpenSSH 7.2", "80": "Apache 2.4.29"},
            "10.10.10.15": {"21": "vsftpd 3.0.3"}
        },
        "vulnerabilities": ["CVE-2017-5638", "CVE-2019-0708"],
        "verbose_logs": "Very long string with detailed logs"
    }
    
    # Optimize for LLM prompt
    optimized = ContextEncoder.optimize_for_llm_prompt(sample_context, max_chars=300)
    print(optimized)
