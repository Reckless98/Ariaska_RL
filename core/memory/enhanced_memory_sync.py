#!/usr/bin/env python3
# core/memory/enhanced_memory_sync.py — ARIASKA Enhanced Memory Synchronization v2.0
# 🧠 Multi-Agent Memory Fusion | 🔄 Real-time Sync | 📊 Knowledge Sharing

import time
import json
import threading
import logging
from typing import Dict, List, Any, Optional, Set
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.gpt_manager import GPTManager
from core.multiagent.memory_router import MemoryRouter

logger = logging.getLogger("ariaska.enhanced_memory_sync")

@dataclass
class MemoryInsight:
    """Structured memory insight for cross-agent sharing."""
    agent_id: str
    timestamp: float
    insight_type: str  # "tactical", "strategic", "environmental", "coordination"
    content: str
    confidence: float
    context: Dict[str, Any]
    relevance_tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryInsight':
        return cls(**data)

class EnhancedMemorySync:
    """
    Enhanced memory synchronization system for multi-agent coordination.
    
    Features:
    - Real-time memory sharing between agents
    - GPT-4o-mini powered insight generation
    - Intelligent relevance filtering
    - Cross-agent knowledge fusion
    - Performance optimization
    - Conflict resolution
    """
    
    def __init__(self, agents: Dict[str, Any], sync_interval: float = 5.0):
        self.agents = agents
        self.sync_interval = sync_interval
        self.gpt_manager = GPTManager.get_instance()
        self.memory_router = MemoryRouter()
        
        # Memory stores
        self.shared_insights: Dict[str, List[MemoryInsight]] = defaultdict(list)
        self.agent_memories: Dict[str, deque] = {
            agent_id: deque(maxlen=1000) for agent_id in agents.keys()
        }
        self.global_knowledge_base: List[MemoryInsight] = []
        
        # Synchronization tracking
        self.last_sync_time = time.time()
        self.sync_lock = threading.Lock()
        self.sync_stats = {
            "total_syncs": 0,
            "insights_shared": 0,
            "conflicts_resolved": 0,
            "knowledge_fusions": 0
        }
        
        # Performance optimization
        self.relevance_cache: Dict[str, Set[str]] = {}
        self.insight_cache: Dict[str, MemoryInsight] = {}
        
        # Background sync thread
        self.sync_thread = None
        self.sync_active = False
        
        logger.info("🧠 Enhanced Memory Sync initialized for multi-agent coordination")
    
    def start_background_sync(self):
        """Start background memory synchronization."""
        if self.sync_thread and self.sync_thread.is_alive():
            return
        
        self.sync_active = True
        self.sync_thread = threading.Thread(target=self._background_sync_loop, daemon=True)
        self.sync_thread.start()
        logger.info("🔄 Background memory sync started")
    
    def stop_background_sync(self):
        """Stop background memory synchronization."""
        self.sync_active = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5.0)
        logger.info("⏹️ Background memory sync stopped")
    
    def _background_sync_loop(self):
        """Background synchronization loop."""
        while self.sync_active:
            try:
                current_time = time.time()
                if current_time - self.last_sync_time >= self.sync_interval:
                    self.synchronize_all_agents()
                    self.last_sync_time = current_time
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Background sync error: {e}")
                time.sleep(5.0)  # Wait before retrying
    
    def add_insight(self, agent_id: str, insight_type: str, content: str, 
                   context: Optional[Dict[str, Any]] = None, 
                   confidence: float = 0.8) -> MemoryInsight:
        """
        Add a new insight from an agent.
        
        Args:
            agent_id: ID of the agent providing the insight
            insight_type: Type of insight (tactical, strategic, etc.)
            content: The insight content
            context: Additional context information
            confidence: Confidence level (0.0 to 1.0)
            
        Returns:
            MemoryInsight: The created insight object
        """
        # Generate relevance tags using GPT
        relevance_tags = self._generate_relevance_tags(content, insight_type)
        
        insight = MemoryInsight(
            agent_id=agent_id,
            timestamp=time.time(),
            insight_type=insight_type,
            content=content,
            confidence=confidence,
            context=context or {},
            relevance_tags=relevance_tags
        )
        
        # Store in agent memory
        self.agent_memories[agent_id].append(insight)
        
        # Add to shared insights
        with self.sync_lock:
            self.shared_insights[agent_id].append(insight)
            self.global_knowledge_base.append(insight)
            
            # Limit global knowledge base size
            if len(self.global_knowledge_base) > 5000:
                self.global_knowledge_base = self.global_knowledge_base[-4000:]
        
        # Cache insight
        insight_key = f"{agent_id}_{insight.timestamp}"
        self.insight_cache[insight_key] = insight
        
        logger.debug(f"💡 Added insight from {agent_id}: {content[:50]}...")
        return insight
    
    def _generate_relevance_tags(self, content: str, insight_type: str) -> List[str]:
        """Generate relevance tags for an insight using GPT-4o-mini."""
        try:
            prompt = f"""
            Analyze this {insight_type} insight and generate 3-5 relevant tags for categorization:
            
            Insight: {content}
            
            Generate tags that would help other agents find this insight when they need related information.
            Focus on: actions, techniques, targets, phases, and contexts.
            Return only the tags separated by commas.
            """
            
            response = self.gpt_manager.gpt_request(
                prompt,
                task_type="analysis",
                agent_id="MemorySync",
                max_tokens=100
            )
            
            if response:
                # Parse tags from response
                tags = [tag.strip().lower() for tag in response.split(',')]
                tags = [tag for tag in tags if tag and len(tag) > 2][:5]  # Max 5 tags
                return tags
            
        except Exception as e:
            logger.warning(f"Failed to generate relevance tags: {e}")
        
        # Fallback tags based on insight type
        fallback_tags = {
            "tactical": ["command", "execution", "immediate"],
            "strategic": ["planning", "coordination", "long-term"],
            "environmental": ["target", "context", "situation"],
            "coordination": ["multi-agent", "cooperation", "sync"]
        }
        return fallback_tags.get(insight_type, ["general"])
    
    def get_relevant_insights(self, agent_id: str, query: str, 
                            max_results: int = 10) -> List[MemoryInsight]:
        """
        Get insights relevant to a query for a specific agent.
        
        Args:
            agent_id: Requesting agent ID
            query: Query string or context
            max_results: Maximum number of results to return
            
        Returns:
            List[MemoryInsight]: Relevant insights ranked by relevance
        """
        # Generate query tags
        query_tags = self._generate_relevance_tags(query, "query")
        
        # Find matching insights
        relevant_insights = []
        
        with self.sync_lock:
            for insight in self.global_knowledge_base:
                # Skip own insights unless specifically requested
                if insight.agent_id == agent_id:
                    continue
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance(insight, query_tags, query)
                
                if relevance_score > 0.3:  # Minimum relevance threshold
                    relevant_insights.append((insight, relevance_score))
        
        # Sort by relevance and return top results
        relevant_insights.sort(key=lambda x: x[1], reverse=True)
        return [insight for insight, score in relevant_insights[:max_results]]
    
    def _calculate_relevance(self, insight: MemoryInsight, query_tags: List[str], 
                           query: str) -> float:
        """Calculate relevance score between insight and query."""
        score = 0.0
        
        # Tag overlap score (40% weight)
        tag_overlap = len(set(insight.relevance_tags) & set(query_tags))
        if insight.relevance_tags:
            tag_score = tag_overlap / len(insight.relevance_tags)
            score += 0.4 * tag_score
        
        # Content similarity score (40% weight)
        content_words = set(insight.content.lower().split())
        query_words = set(query.lower().split())
        if content_words and query_words:
            content_overlap = len(content_words & query_words)
            content_score = content_overlap / len(content_words | query_words)
            score += 0.4 * content_score
        
        # Confidence score (10% weight)
        score += 0.1 * insight.confidence
        
        # Recency score (10% weight)
        time_factor = max(0, 1 - (time.time() - insight.timestamp) / 3600)  # 1 hour decay
        score += 0.1 * time_factor
        
        return score
    
    def synchronize_all_agents(self):
        """Synchronize memories between all agents."""
        try:
            with self.sync_lock:
                sync_start_time = time.time()
                
                # Gather new insights from all agents
                new_insights = []
                for agent_id, agent in self.agents.items():
                    if hasattr(agent, 'get_recent_insights'):
                        try:
                            recent = agent.get_recent_insights()
                            for insight_data in recent:
                                insight = self._convert_to_memory_insight(agent_id, insight_data)
                                if insight:
                                    new_insights.append(insight)
                        except Exception as e:
                            logger.warning(f"Failed to get insights from {agent_id}: {e}")
                
                # Process new insights
                for insight in new_insights:
                    self.global_knowledge_base.append(insight)
                    self.sync_stats["insights_shared"] += 1
                
                # Perform knowledge fusion
                if len(new_insights) > 1:
                    self._perform_knowledge_fusion(new_insights)
                
                # Distribute relevant insights to agents
                self._distribute_insights_to_agents()
                
                # Update sync stats
                self.sync_stats["total_syncs"] += 1
                sync_duration = time.time() - sync_start_time
                
                logger.info(f"🔄 Memory sync completed in {sync_duration:.3f}s - {len(new_insights)} new insights")
                
        except Exception as e:
            logger.error(f"Memory synchronization failed: {e}")
    
    def _convert_to_memory_insight(self, agent_id: str, insight_data: Any) -> Optional[MemoryInsight]:
        """Convert agent insight data to MemoryInsight format."""
        try:
            if isinstance(insight_data, dict):
                return MemoryInsight(
                    agent_id=agent_id,
                    timestamp=insight_data.get('timestamp', time.time()),
                    insight_type=insight_data.get('type', 'general'),
                    content=insight_data.get('content', ''),
                    confidence=insight_data.get('confidence', 0.8),
                    context=insight_data.get('context', {}),
                    relevance_tags=insight_data.get('tags', [])
                )
            elif isinstance(insight_data, str):
                return MemoryInsight(
                    agent_id=agent_id,
                    timestamp=time.time(),
                    insight_type='general',
                    content=insight_data,
                    confidence=0.8,
                    context={},
                    relevance_tags=self._generate_relevance_tags(insight_data, 'general')
                )
        except Exception as e:
            logger.warning(f"Failed to convert insight data: {e}")
        return None
    
    def _perform_knowledge_fusion(self, insights: List[MemoryInsight]):
        """Perform knowledge fusion using GPT-4o-mini to combine related insights."""
        try:
            if len(insights) < 2:
                return
            
            # Group insights by similarity
            insight_groups = self._group_similar_insights(insights)
            
            for group in insight_groups:
                if len(group) >= 2:
                    # Generate fused knowledge
                    fused_insight = self._fuse_insight_group(group)
                    if fused_insight:
                        self.global_knowledge_base.append(fused_insight)
                        self.sync_stats["knowledge_fusions"] += 1
                        
        except Exception as e:
            logger.warning(f"Knowledge fusion failed: {e}")
    
    def _group_similar_insights(self, insights: List[MemoryInsight]) -> List[List[MemoryInsight]]:
        """Group similar insights together for fusion."""
        groups = []
        used_insights = set()
        
        for i, insight1 in enumerate(insights):
            if i in used_insights:
                continue
            
            group = [insight1]
            used_insights.add(i)
            
            for j, insight2 in enumerate(insights[i+1:], i+1):
                if j in used_insights:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_insight_similarity(insight1, insight2)
                if similarity > 0.6:  # Similarity threshold
                    group.append(insight2)
                    used_insights.add(j)
            
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def _calculate_insight_similarity(self, insight1: MemoryInsight, insight2: MemoryInsight) -> float:
        """Calculate similarity between two insights."""
        # Tag overlap
        tags1 = set(insight1.relevance_tags)
        tags2 = set(insight2.relevance_tags)
        tag_similarity = len(tags1 & tags2) / len(tags1 | tags2) if tags1 | tags2 else 0
        
        # Content similarity (simple word overlap)
        words1 = set(insight1.content.lower().split())
        words2 = set(insight2.content.lower().split())
        content_similarity = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
        
        # Type similarity
        type_similarity = 1.0 if insight1.insight_type == insight2.insight_type else 0.5
        
        return 0.4 * tag_similarity + 0.4 * content_similarity + 0.2 * type_similarity
    
    def _fuse_insight_group(self, insights: List[MemoryInsight]) -> Optional[MemoryInsight]:
        """Fuse a group of related insights using GPT-4o-mini."""
        try:
            insight_texts = [f"{insight.agent_id}: {insight.content}" for insight in insights]
            context = "\n".join(insight_texts)
            
            prompt = f"""
            Fuse these related insights from different agents into a single comprehensive insight:
            
            {context}
            
            Create a unified insight that combines the key information while avoiding redundancy.
            Focus on actionable intelligence and cross-agent learnings.
            """
            
            fused_content = self.gpt_manager.gpt_request(
                prompt,
                task_type="synthesis",
                agent_id="MemorySync",
                max_tokens=300
            )
            
            if fused_content:
                # Create fused insight
                all_tags = []
                for insight in insights:
                    all_tags.extend(insight.relevance_tags)
                unique_tags = list(set(all_tags))
                
                avg_confidence = sum(insight.confidence for insight in insights) / len(insights)
                
                return MemoryInsight(
                    agent_id="FUSED",
                    timestamp=time.time(),
                    insight_type="fused",
                    content=fused_content,
                    confidence=avg_confidence,
                    context={"source_agents": [insight.agent_id for insight in insights]},
                    relevance_tags=unique_tags[:5]  # Limit to 5 tags
                )
                
        except Exception as e:
            logger.warning(f"Insight fusion failed: {e}")
        
        return None
    
    def _distribute_insights_to_agents(self):
        """Distribute relevant insights to each agent."""
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'receive_shared_insights'):
                try:
                    # Get agent's current context/interests
                    agent_context = getattr(agent, 'current_context', 'general operations')
                    
                    # Find relevant insights
                    relevant = self.get_relevant_insights(agent_id, agent_context, max_results=5)
                    
                    if relevant:
                        # Send to agent
                        agent.receive_shared_insights([insight.to_dict() for insight in relevant])
                        
                except Exception as e:
                    logger.warning(f"Failed to distribute insights to {agent_id}: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory synchronization statistics."""
        with self.sync_lock:
            stats = {
                "sync_stats": dict(self.sync_stats),
                "memory_sizes": {
                    agent_id: len(memory) 
                    for agent_id, memory in self.agent_memories.items()
                },
                "global_knowledge_size": len(self.global_knowledge_base),
                "shared_insights_size": sum(len(insights) for insights in self.shared_insights.values()),
                "cache_sizes": {
                    "relevance_cache": len(self.relevance_cache),
                    "insight_cache": len(self.insight_cache)
                },
                "last_sync_time": self.last_sync_time,
                "sync_active": self.sync_active
            }
        return stats
    
    def cleanup_old_memories(self, max_age_hours: float = 24.0):
        """Clean up old memories to maintain performance."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        with self.sync_lock:
            # Clean global knowledge base
            self.global_knowledge_base = [
                insight for insight in self.global_knowledge_base
                if insight.timestamp > cutoff_time
            ]
            
            # Clean shared insights
            for agent_id in self.shared_insights:
                self.shared_insights[agent_id] = [
                    insight for insight in self.shared_insights[agent_id]
                    if insight.timestamp > cutoff_time
                ]
            
            # Clean caches
            old_keys = [
                key for key, insight in self.insight_cache.items()
                if insight.timestamp <= cutoff_time
            ]
            for key in old_keys:
                del self.insight_cache[key]
        
        logger.info(f"🧹 Cleaned {len(old_keys)} old memories (>{max_age_hours}h)")

# Convenience function
def create_enhanced_memory_sync(agents: Dict[str, Any], sync_interval: float = 5.0) -> EnhancedMemorySync:
    """Create and return an enhanced memory sync instance."""
    return EnhancedMemorySync(agents, sync_interval)
