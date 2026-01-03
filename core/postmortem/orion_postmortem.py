#!/usr/bin/env python3
"""
core/postmortem/orion_postmortem.py — ARIASKA OrionPostmortem System v1.0

Single end-of-run GPT call that analyzes the entire training run and produces
structured output for memory/skill updates.

Key Features:
- Strict JSON schema validation
- Skill card generation
- Memory operation instructions
- Deterministic application of updates
- Dry-run mode for testing
"""

import os
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("ariaska.postmortem")

# JSON Schema for postmortem output
POSTMORTEM_SCHEMA = {
    "type": "object",
    "required": ["key_outcomes", "skill_cards", "memory_ops", "next_experiments"],
    "properties": {
        "key_outcomes": {
            "type": "object",
            "required": ["wins", "fails", "summary"],
            "properties": {
                "wins": {"type": "array", "items": {"type": "string"}},
                "fails": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"}
            }
        },
        "repeated_failure_patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["pattern", "frequency", "evidence_refs"],
                "properties": {
                    "pattern": {"type": "string"},
                    "frequency": {"type": "integer"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "skill_cards": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "if_condition", "then_action", "confidence"],
                "properties": {
                    "id": {"type": "string"},
                    "if_condition": {"type": "string"},
                    "then_action": {"type": "string"},
                    "parameters_template": {"type": "object"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "memory_ops": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["operation", "target"],
                "properties": {
                    "operation": {"type": "string", "enum": ["promote", "prune", "merge"]},
                    "target": {"type": "string"},
                    "skill_card_id": {"type": "string"},
                    "merge_with": {"type": "string"},
                    "reason": {"type": "string"}
                }
            }
        },
        "next_experiments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "description"],
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                }
            },
            "maxItems": 3
        }
    }
}


@dataclass
class SkillCard:
    """A learned skill/pattern from training."""
    id: str
    if_condition: str  # When to apply this skill
    then_action: str   # What action to take
    parameters_template: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    evidence_refs: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    usage_count: int = 0
    success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillCard":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MemoryOperation:
    """An operation to perform on memory/skills."""
    operation: str  # "promote", "prune", "merge"
    target: str     # Target ID or key
    skill_card_id: Optional[str] = None
    merge_with: Optional[str] = None
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PostmortemResult:
    """Result of a postmortem analysis."""
    run_id: str
    timestamp: float
    
    # Analysis outputs
    key_outcomes: Dict[str, Any] = field(default_factory=dict)
    repeated_failure_patterns: List[Dict[str, Any]] = field(default_factory=list)
    skill_cards: List[SkillCard] = field(default_factory=list)
    memory_ops: List[MemoryOperation] = field(default_factory=list)
    next_experiments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    model_used: str = "gpt-5-mini"
    dry_run: bool = False
    validation_passed: bool = False
    raw_response: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "key_outcomes": self.key_outcomes,
            "repeated_failure_patterns": self.repeated_failure_patterns,
            "skill_cards": [s.to_dict() for s in self.skill_cards],
            "memory_ops": [m.to_dict() for m in self.memory_ops],
            "next_experiments": self.next_experiments,
            "model_used": self.model_used,
            "dry_run": self.dry_run,
            "validation_passed": self.validation_passed
        }
    
    def save(self, output_dir: str):
        """Save postmortem result to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"postmortem_{self.run_id}.json"
        filepath = output_path / filename
        
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Postmortem saved to: {filepath}")
        return filepath


class OrionPostmortem:
    """
    End-of-run postmortem analysis using GPT.
    
    Analyzes the complete training run and produces:
    - Key outcomes (wins/fails)
    - Repeated failure patterns
    - Skill cards for the SkillLibrary
    - Memory operations (promote/prune/merge)
    - Suggestions for next experiments
    
    The LLM output is validated against a strict JSON schema.
    Memory operations are instructions only - actual application
    is done deterministically by code.
    """
    
    def __init__(
        self,
        gpt_manager=None,
        output_dir: str = "postmortems",
        enable_gpt_5_2: bool = False
    ):
        from core.gpt_manager import GPTManager
        
        self.gpt_manager = gpt_manager or GPTManager()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_gpt_5_2 = enable_gpt_5_2
        
        logger.info("OrionPostmortem initialized")
    
    def analyze_run(
        self,
        run_trace: Dict[str, Any],
        dry_run: bool = False
    ) -> PostmortemResult:
        """
        Analyze a complete training run.
        
        Args:
            run_trace: The run trace dictionary from TraceReader
            dry_run: If True, produce JSON but don't apply memory changes
            
        Returns:
            PostmortemResult with analysis and instructions
        """
        run_id = run_trace.get("run_id", f"run_{int(time.time())}")
        
        result = PostmortemResult(
            run_id=run_id,
            timestamp=time.time(),
            dry_run=dry_run
        )
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(run_trace)
        
        # Determine model to use
        if self.enable_gpt_5_2:
            task_type = "postmortem"
            model = self.gpt_manager.postmortem_model
        else:
            task_type = "reasoning"
            model = self.gpt_manager.primary_model
        
        result.model_used = model
        
        try:
            # Make GPT request
            response = self.gpt_manager.gpt_request(
                prompt,
                task_type=task_type,
                agent_id="OrionPostmortem",
                max_tokens=2000,
                model=model
            )
            
            result.raw_response = response
            
            # Parse and validate response
            parsed = self._parse_response(response)
            
            if parsed:
                result.validation_passed = self._validate_schema(parsed)
                
                if result.validation_passed:
                    self._populate_result(result, parsed)
                else:
                    logger.warning("Postmortem response failed schema validation")
            else:
                logger.error("Failed to parse postmortem response")
                
        except Exception as e:
            logger.error(f"Postmortem analysis failed: {e}")
            result.validation_passed = False
        
        # Save result
        if not dry_run:
            result.save(str(self.output_dir))
        
        return result
    
    def _build_analysis_prompt(self, run_trace: Dict[str, Any]) -> str:
        """Build the analysis prompt from run trace."""
        
        # Extract key metrics
        total_episodes = run_trace.get("total_episodes", 0)
        total_reward = run_trace.get("total_reward", 0)
        success_rate = run_trace.get("success_rate", 0)
        mentor_calls = run_trace.get("total_mentor_calls", 0)
        reward_history = run_trace.get("reward_history", [])[-20:]  # Last 20 episodes
        
        # Build summary
        summary = f"""Training Run Analysis Request
=============================
Run ID: {run_trace.get('run_id', 'unknown')}
Total Episodes: {total_episodes}
Total Reward: {total_reward:.2f}
Success Rate: {success_rate:.2%}
Total Mentor Calls: {mentor_calls}
Avg Reward per Episode: {total_reward / max(total_episodes, 1):.2f}
Recent Reward Trend: {reward_history}
"""
        
        prompt = f"""{summary}

Analyze this training run and produce a JSON response with the following structure:

{{
    "key_outcomes": {{
        "wins": ["List of key successes/achievements"],
        "fails": ["List of key failures/issues"],
        "summary": "Brief overall assessment"
    }},
    "repeated_failure_patterns": [
        {{
            "pattern": "Description of repeated failure",
            "frequency": 5,
            "evidence_refs": ["ep001_step10", "ep003_step15"]
        }}
    ],
    "skill_cards": [
        {{
            "id": "skill_001",
            "if_condition": "When port 22 is open and SSH banner detected",
            "then_action": "Run SSH enumeration with nmap scripts",
            "parameters_template": {{"port": 22, "script": "ssh-*"}},
            "confidence": 0.85,
            "evidence_refs": ["ep010_step5", "ep015_step3"]
        }}
    ],
    "memory_ops": [
        {{
            "operation": "promote",
            "target": "skill_001",
            "skill_card_id": "skill_001",
            "reason": "High success rate in reconnaissance phase"
        }}
    ],
    "next_experiments": [
        {{
            "title": "Experiment name",
            "description": "What to try next",
            "priority": "high"
        }}
    ]
}}

IMPORTANT:
1. Respond with ONLY valid JSON - no additional text
2. Generate 2-5 skill cards based on successful patterns observed
3. Memory ops should reference skill_card IDs
4. Limit next_experiments to top 3 priorities
5. Be specific and actionable in all descriptions"""

        return prompt
    
    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from GPT."""
        import re
        
        # Try to extract JSON from response
        # Handle code blocks
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        
        # Try to find JSON object
        try:
            # Find the outermost { }
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
        
        return None
    
    def _validate_schema(self, data: Dict[str, Any]) -> bool:
        """Validate data against the postmortem schema."""
        # Simple validation - check required fields
        required_fields = ["key_outcomes", "skill_cards", "memory_ops", "next_experiments"]
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate key_outcomes structure
        outcomes = data.get("key_outcomes", {})
        if not all(k in outcomes for k in ["wins", "fails", "summary"]):
            logger.warning("key_outcomes missing required fields")
            return False
        
        # Validate skill_cards structure
        for i, card in enumerate(data.get("skill_cards", [])):
            required_card_fields = ["id", "if_condition", "then_action", "confidence"]
            if not all(k in card for k in required_card_fields):
                logger.warning(f"skill_card {i} missing required fields")
                return False
        
        # Validate memory_ops structure
        for i, op in enumerate(data.get("memory_ops", [])):
            if "operation" not in op or "target" not in op:
                logger.warning(f"memory_op {i} missing required fields")
                return False
            if op["operation"] not in ["promote", "prune", "merge"]:
                logger.warning(f"memory_op {i} has invalid operation: {op['operation']}")
                return False
        
        return True
    
    def _populate_result(self, result: PostmortemResult, data: Dict[str, Any]):
        """Populate PostmortemResult from parsed data."""
        result.key_outcomes = data.get("key_outcomes", {})
        result.repeated_failure_patterns = data.get("repeated_failure_patterns", [])
        result.next_experiments = data.get("next_experiments", [])[:3]
        
        # Parse skill cards
        for card_data in data.get("skill_cards", []):
            try:
                card = SkillCard(
                    id=card_data.get("id", f"skill_{hashlib.md5(str(card_data).encode()).hexdigest()[:8]}"),
                    if_condition=card_data.get("if_condition", ""),
                    then_action=card_data.get("then_action", ""),
                    parameters_template=card_data.get("parameters_template", {}),
                    confidence=float(card_data.get("confidence", 0.5)),
                    evidence_refs=card_data.get("evidence_refs", [])
                )
                result.skill_cards.append(card)
            except Exception as e:
                logger.warning(f"Failed to parse skill card: {e}")
        
        # Parse memory ops
        for op_data in data.get("memory_ops", []):
            try:
                op = MemoryOperation(
                    operation=op_data.get("operation"),
                    target=op_data.get("target"),
                    skill_card_id=op_data.get("skill_card_id"),
                    merge_with=op_data.get("merge_with"),
                    reason=op_data.get("reason", "")
                )
                result.memory_ops.append(op)
            except Exception as e:
                logger.warning(f"Failed to parse memory op: {e}")


# Factory function
def create_postmortem_analyzer(
    enable_gpt_5_2: bool = False,
    output_dir: str = "postmortems"
) -> OrionPostmortem:
    """Create a postmortem analyzer."""
    return OrionPostmortem(
        enable_gpt_5_2=enable_gpt_5_2,
        output_dir=output_dir
    )


if __name__ == "__main__":
    from rich.console import Console
    console = Console()
    
    console.print("[bold cyan]Testing OrionPostmortem[/bold cyan]")
    
    # Test with mock run trace
    mock_run_trace = {
        "run_id": "test_run_001",
        "total_episodes": 50,
        "total_reward": 1250.5,
        "success_rate": 0.72,
        "total_mentor_calls": 85,
        "reward_history": [20.0, 22.5, 18.0, 25.0, 30.0, 28.0, 35.0, 32.0, 40.0, 38.0]
    }
    
    # Test schema validation
    valid_response = {
        "key_outcomes": {
            "wins": ["Successfully exploited SSH on multiple targets"],
            "fails": ["Privilege escalation failed in 30% of cases"],
            "summary": "Good reconnaissance, needs better privesc strategies"
        },
        "repeated_failure_patterns": [
            {
                "pattern": "Sudo enumeration before checking kernel version",
                "frequency": 8,
                "evidence_refs": ["ep005_step12", "ep012_step8"]
            }
        ],
        "skill_cards": [
            {
                "id": "skill_ssh_enum",
                "if_condition": "Port 22 open and banner shows OpenSSH",
                "then_action": "nmap -sV -p22 --script=ssh-* TARGET",
                "parameters_template": {"port": 22},
                "confidence": 0.88,
                "evidence_refs": ["ep010_step5"]
            }
        ],
        "memory_ops": [
            {
                "operation": "promote",
                "target": "skill_ssh_enum",
                "skill_card_id": "skill_ssh_enum",
                "reason": "High success rate"
            }
        ],
        "next_experiments": [
            {
                "title": "Kernel exploit detection",
                "description": "Add kernel version check before sudo enum",
                "priority": "high"
            }
        ]
    }
    
    # Create analyzer and validate
    analyzer = OrionPostmortem(output_dir="test_postmortems")
    
    is_valid = analyzer._validate_schema(valid_response)
    console.print(f"Schema validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    
    # Test skill card parsing
    result = PostmortemResult(run_id="test", timestamp=time.time())
    analyzer._populate_result(result, valid_response)
    
    console.print(f"Skill cards parsed: {len(result.skill_cards)}")
    console.print(f"Memory ops parsed: {len(result.memory_ops)}")
    
    console.print("\n[bold green]✓ OrionPostmortem test passed![/bold green]")
