import os
import time
import json
import hashlib
import random
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from enum import Enum
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.utils.local_llm_manager import LocalLLMManager
from core.gpt_manager import GPTManager

console = Console()

class LLMModelTier(Enum):
    """Enumeration of LLM tiers for fallback chain"""
    TIER1_LOCAL_SMALL = 1   # Lily - Fast, lightweight local model
    TIER2_LOCAL_MEDIUM = 2  # Seneca - More capable local model
    TIER3_CLOUD_SMALL = 3   # GPT-4.1-nano - Small cloud model
    TIER4_CLOUD_LARGE = 4   # GPT-4o-mini - Primary capable model

class CommandOutput(BaseModel):
    """Base schema for structured LLM command outputs"""
    command: str
    confidence: float = 0.8
    reasoning: Optional[str] = None
    
    class Config:
        extra = "forbid"

class ScanCommand(CommandOutput):
    """Schema for scan commands"""
    targets: List[str]
    ports: Optional[List[int]] = None
    scan_type: str = "tcp"
    stealth_level: int = 1

class ExploitCommand(CommandOutput):
    """Schema for exploit commands"""
    target: str
    service: str
    exploit_type: str
    payload: Optional[str] = None
    risk_level: int = 2

class LLMOrchestrator:
    """
    Centralized LLM orchestration system with intelligent routing, fallbacks, and token management.
    
    Implements a fallback chain: Local Small → Local Medium → Cloud Small → Cloud Large
    
    Features:
    - Automatic model selection based on task complexity
    - Token usage tracking by agent and task
    - Prompt caching (in-memory and persistent)
    - Command validation with Pydantic schemas
    - Intelligent prompt optimization
    """
    
    # Default models for different tasks
    DEFAULT_MODELS = {
        "reasoning": "gpt-4o-mini",
        "decision": "gpt-4o-mini",
        "embedding": "gpt-4.1-nano",
        "classification": "gpt-4.1-nano",
        "tactical": "gpt-4o-mini",
        "strategic": "gpt-4o-mini",
        "fallback": "gpt-4o-mini",
    }
    
    # Token limits for different models
    TOKEN_LIMITS = {
        "gpt-4o-mini": 8000,
        "gpt-4o-mini": 128000,
        "gpt-4.1-nano": 4000,
        "claude-3-opus": 180000,
        "mistral-large": 32000,
    }
    
    # Prompt templates for different tasks
    PROMPT_TEMPLATES = {
        "scan": """Using only the provided context, suggest {num_targets} optimal scanning targets.
Context: {context}

Output a JSON object with the following structure:
{
  "command": "scan",
  "targets": ["target1", "target2"],
  "ports": [22, 80, 443],
  "scan_type": "syn",
  "stealth_level": 2,
  "confidence": 0.9,
  "reasoning": "brief explanation"
}""",
        
        "exploit": """Based on scan results, suggest the best exploit approach.
Context: {context}

Output a JSON object with the following structure:
{
  "command": "exploit",
  "target": "target_ip_or_service",
  "service": "service_name",
  "exploit_type": "exploit_name",
  "payload": "optional_payload",
  "risk_level": 1-5,
  "confidence": 0.1-1.0,
  "reasoning": "brief explanation"
}""",
        
        "strategic": """Analyze the current mission state and suggest a strategic approach.
Context: {context}

Respond with a concise strategic recommendation in 3 bullet points or less."""
    }
    
    def __init__(self, 
                 cache_dir: str = "cache/llm_responses", 
                 tracking_enabled: bool = True,
                 max_retries: int = 3):
        """
        Initialize the LLM orchestrator.
        
        Args:
            cache_dir: Directory for caching LLM responses
            tracking_enabled: Whether to track token usage
            max_retries: Maximum retries for each model tier
        """
        self.cache_dir = cache_dir
        self.tracking_enabled = tracking_enabled
        self.max_retries = max_retries
        self.in_memory_cache = {}
        self.token_usage = {}
        self.agent_usage = {}
        self.task_usage = {}
        self.fallback_counts = {}
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize local models
        try:
            self.lily = LocalLLMManager(model_name=os.environ.get(
                "ARIASKA_LILY_MODEL", "QuantFactory/Lily-Cybersecurity-7B-v0.2-GGUF:Q8_0"))
            console.print("[green]✓ Lily LLM initialized[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not initialize Lily LLM: {e}[/yellow]")
            self.lily = None
            
        try:
            self.seneca = LocalLLMManager(model_name=os.environ.get(
                "ARIASKA_SENECA_MODEL", "wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF"))
            console.print("[green]✓ Seneca LLM initialized[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not initialize Seneca LLM: {e}[/yellow]")
            self.seneca = None
        
        # Initialize GPT Manager
        try:
            self.gpt = GPTManager()
            console.print("[green]✓ GPT Manager initialized[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not initialize GPT Manager: {e}[/yellow]")
            self.gpt = None
            
        # Load cache from disk if available
        self._load_cache()
        
        console.print(f"[green]✓ LLM Orchestrator initialized (cache: {len(self.in_memory_cache)} items)[/green]")
        
    def _load_cache(self):
        """Load LLM response cache from disk."""
        cache_file = os.path.join(self.cache_dir, "response_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self.in_memory_cache = json.load(f)
                console.print(f"[green]✓ Loaded {len(self.in_memory_cache)} cached responses[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ Failed to load cache: {e}[/yellow]")
                self.in_memory_cache = {}
                
    def _save_cache(self):
        """Save LLM response cache to disk."""
        cache_file = os.path.join(self.cache_dir, "response_cache.json")
        try:
            with open(cache_file, "w") as f:
                json.dump(self.in_memory_cache, f)
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to save cache: {e}[/yellow]")
            
    def _get_cache_key(self, prompt: str, model: str) -> str:
        """
        Generate a cache key for a prompt and model.
        
        Args:
            prompt: The prompt to generate a key for
            model: The model to generate a key for
            
        Returns:
            Cache key string
        """
        hash_input = f"{prompt}|{model}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()
        
    def _track_tokens(self, agent_id: Optional[str], task_type: str, model: str, tokens: int):
        """
        Track token usage by agent, task type, and model.
        
        Args:
            agent_id: Agent using the tokens (optional)
            task_type: Type of task
            model: Model used
            tokens: Number of tokens used
        """
        if not self.tracking_enabled:
            return
            
        # Track by model
        self.token_usage.setdefault(model, 0)
        self.token_usage[model] += tokens
        
        # Track by agent if provided
        if agent_id:
            self.agent_usage.setdefault(agent_id, {})
            self.agent_usage[agent_id].setdefault(model, 0)
            self.agent_usage[agent_id][model] += tokens
            
        # Track by task type
        self.task_usage.setdefault(task_type, 0)
        self.task_usage[task_type] += tokens
    
    def route_task(self, 
                  task_type: str, 
                  prompt: str, 
                  agent_id: Optional[str] = None,
                  use_cache: bool = True,
                  schema: Optional[BaseModel] = None,
                  **kwargs) -> str:
        """
        Unified method for routing a task to appropriate LLMs with fallback chain.
        
        Args:
            task_type: Type of task (planner, tactical, strategic, etc.)
            prompt: The prompt to send
            agent_id: Agent ID for tracking
            use_cache: Whether to use cache
            schema: Pydantic schema for validating output (optional)
            
        Returns:
            Response from LLM
        """
        # Check cache first if enabled
        if use_cache:
            cache_key = f"{task_type}|{prompt.strip()[:120]}"
            if cache_key in self.in_memory_cache:
                return self.in_memory_cache[cache_key]
        
        t0 = time.time()
        
        # Determine the appropriate sequence of models to try based on task_type
        if task_type == "tactical" or len(prompt) < 120:
            # For tactical/simple tasks, start with Lily (smallest model)
            model_sequence = [
                (LLMModelTier.TIER1_LOCAL_SMALL, "lily"),
                (LLMModelTier.TIER2_LOCAL_MEDIUM, "seneca"),
                (LLMModelTier.TIER4_CLOUD_LARGE, "gpt")
            ]
        elif task_type == "planner":
            # For planning tasks, start with Seneca
            model_sequence = [
                (LLMModelTier.TIER2_LOCAL_MEDIUM, "seneca"),
                (LLMModelTier.TIER4_CLOUD_LARGE, "gpt")
            ]
        elif task_type == "strategic" or len(prompt) > 300:
            # For complex strategic tasks, go straight to GPT
            model_sequence = [
                (LLMModelTier.TIER4_CLOUD_LARGE, "gpt")
            ]
        else:
            # Default fallback chain
            model_sequence = [
                (LLMModelTier.TIER1_LOCAL_SMALL, "lily"),
                (LLMModelTier.TIER2_LOCAL_MEDIUM, "seneca"),
                (LLMModelTier.TIER4_CLOUD_LARGE, "gpt")
            ]
            
        # Try each model in sequence until we get a valid response
        response = None
        last_error = None
        model_used = None
        
        for tier, model_name in model_sequence:
            if response:
                break
                
            # Skip if model not available
            if model_name == "lily" and self.lily is None:
                continue
            if model_name == "seneca" and self.seneca is None:
                continue
            if model_name == "gpt" and self.gpt is None:
                continue
                
            for retry in range(self.max_retries):
                try:
                    if model_name == "lily":
                        lily_prompt = "You are a concise assistant: answer succinctly.\n" + prompt
                        raw_response = self.lily.query(lily_prompt)
                        model_used = "LilyLLM"
                    elif model_name == "seneca":
                        raw_response = self.seneca.query(prompt)
                        model_used = "SenecaLLM"
                    elif model_name == "gpt":
                        if task_type == "dual-llm-feedback":
                            raw_response = self.gpt.dual_llm_feedback(prompt)
                        else:
                            raw_response = self.gpt.smart_decision(task_type, prompt)
                        model_used = "GPT-4o"
                    
                    # Validate response if schema provided
                    if schema is not None:
                        validated_response = self._validate_and_fix_output(raw_response, schema, model_name, prompt)
                        if validated_response:
                            response = validated_response
                            break
                    else:
                        # Basic validation - not empty, not too long, not gibberish
                        if raw_response and len(raw_response) < 800 and raw_response.count(" ") >= 2:
                            response = raw_response
                            break
                        
                except Exception as e:
                    last_error = e
                    console.print(f"[yellow]⚠️ {model_name} error (retry {retry+1}/{self.max_retries}): {str(e)}[/yellow]")
                    
                # If we get here, this model tier failed
                if retry == self.max_retries - 1:
                    console.print(f"[red]❌ {model_name} failed after {self.max_retries} retries[/red]")
                    # Track fallbacks
                    self.fallback_counts.setdefault(model_name, 0)
                    self.fallback_counts[model_name] += 1
        
        # If all models failed
        if response is None:
            error_msg = f"All LLM models failed. Last error: {last_error}"
            console.print(f"[red]{error_msg}[/red]")
            return f"Error: {error_msg}"
        
        # Log the successful response
        elapsed = time.time() - t0
        tokens = self._estimate_tokens(prompt, response)
        self._track_tokens(agent_id, task_type, model_used, tokens)
        
        # Show different colored output based on model used
        if model_used == "LilyLLM":
            console.print(f"[cyan]🌸 {model_used} ({tokens} tokens, {elapsed:.2f}s): {response[:100]}...[/cyan]")
        elif model_used == "SenecaLLM":
            console.print(f"[blue]🦉 {model_used} ({tokens} tokens, {elapsed:.2f}s): {response[:100]}...[/blue]")
        else:
            console.print(f"[magenta]🤖 {model_used} ({tokens} tokens, {elapsed:.2f}s): {response[:100]}...[/magenta]")
        
        # Cache the response
        if use_cache:
            cache_key = f"{task_type}|{prompt.strip()[:120]}"
            self.in_memory_cache[cache_key] = response
            # Periodically save cache to disk (random chance to avoid IO bottleneck)
            if random.random() < 0.1:
                self._save_cache()
                
        return response
    
    def request_strategy(self, context: Dict[str, Any], task_type: str = "tactical", agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Request a strategy from the LLM chain with proper context processing and output validation.
        
        Args:
            context: Context dictionary with relevant information
            task_type: Type of strategy (tactical, strategic, etc.)
            agent_id: Agent ID for tracking
            
        Returns:
            Validated and structured strategy output
        """
        # Preprocess context to reduce token usage
        processed_context = self._preprocess_context(context)
        
        # Select appropriate template and schema
        if task_type == "scan":
            template = self.PROMPT_TEMPLATES["scan"]
            schema = ScanCommand
            num_targets = context.get("num_targets", 3)
            prompt = template.format(context=processed_context, num_targets=num_targets)
        elif task_type == "exploit":
            template = self.PROMPT_TEMPLATES["exploit"]
            schema = ExploitCommand
            prompt = template.format(context=processed_context)
        else:
            template = self.PROMPT_TEMPLATES["strategic"]
            schema = None
            prompt = template.format(context=processed_context)
            
        # Route task through the fallback chain
        response = self.route_task(
            task_type=task_type,
            prompt=prompt,
            agent_id=agent_id,
            use_cache=True,
            schema=schema
        )
        
        # Parse and validate the response
        if schema is not None:
            try:
                # Try to parse JSON response
                response_dict = self._extract_json_from_text(response)
                validated_response = schema(**response_dict)
                return validated_response.dict()
            except (ValidationError, json.JSONDecodeError) as e:
                console.print(f"[yellow]⚠️ Failed to validate response: {e}[/yellow]")
                return {"command": "error", "error": str(e), "raw_response": response}
        else:
            return {"command": "response", "content": response}
    
    def _validate_and_fix_output(self, response: str, schema: BaseModel, model_name: str, original_prompt: str) -> Optional[str]:
        """
        Validate LLM output against schema and try to fix if invalid.
        
        Args:
            response: Raw LLM response
            schema: Pydantic schema to validate against
            model_name: Name of the model that generated the response
            original_prompt: Original prompt sent to the model
            
        Returns:
            Validated response or None if validation failed
        """
        try:
            # Try to extract JSON from response
            json_data = self._extract_json_from_text(response)
            
            # Validate against schema
            validated = schema(**json_data)
            return response
        except (ValidationError, json.JSONDecodeError) as e:
            # If using a local model, don't attempt fixes
            if model_name in ["lily", "seneca"]:
                return None
                
            # For cloud models, try to fix the output
            fix_prompt = f"""The previous response didn't match the required JSON schema. 
Error: {str(e)}

Original prompt: {original_prompt}

Fix the output to match this schema exactly:
{schema.schema_json()}

Output ONLY valid JSON without any explanations or markdown formatting."""
            
            try:
                if model_name == "gpt":
                    fixed_response = self.gpt.smart_decision("fix", fix_prompt)
                    
                    # Verify the fixed response
                    json_data = self._extract_json_from_text(fixed_response)
                    schema(**json_data)  # This will raise ValidationError if still invalid
                    return fixed_response
            except Exception:
                return None
                
            return None
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text that might contain other content.
        
        Args:
            text: Text that might contain JSON
            
        Returns:
            Extracted JSON as dict
        """
        # Look for JSON content between curly braces
        import re
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        else:
            # If no JSON found, try to parse the entire text
            return json.loads(text)
    
    def _preprocess_context(self, context: Dict[str, Any]) -> str:
        """
        Preprocess context to minimize tokens while preserving essential information.
        
        Args:
            context: Raw context dictionary
            
        Returns:
            Preprocessed context string
        """
        # Start with essential information
        essential_keys = ["mission_phase", "discovered_hosts", "discovered_services", "previous_actions"]
        
        # Build concise context string
        context_parts = []
        
        # Add mission phase
        if "mission_phase" in context:
            context_parts.append(f"Phase: {context['mission_phase']}")
            
        # Add discovered hosts and services
        if "discovered_hosts" in context:
            hosts = context["discovered_hosts"]
            if isinstance(hosts, list) and len(hosts) > 0:
                if len(hosts) > 5:
                    # Summarize if too many hosts
                    context_parts.append(f"Hosts: {len(hosts)} discovered including {', '.join(hosts[:3])}...")
                else:
                    context_parts.append(f"Hosts: {', '.join(hosts)}")
                    
        if "discovered_services" in context:
            services = context["discovered_services"]
            if isinstance(services, dict) and len(services) > 0:
                service_parts = []
                for host, host_services in list(services.items())[:3]:  # Limit to first 3 hosts
                    if len(host_services) > 3:
                        service_parts.append(f"{host}: {', '.join(host_services[:3])}...")
                    else:
                        service_parts.append(f"{host}: {', '.join(host_services)}")
                context_parts.append(f"Services: {'; '.join(service_parts)}")
                
        # Add recent actions (limited)
        if "previous_actions" in context:
            actions = context["previous_actions"]
            if isinstance(actions, list) and len(actions) > 0:
                recent_actions = actions[-3:]  # Only most recent 3 actions
                context_parts.append(f"Recent actions: {'; '.join(recent_actions)}")
                
        # Add objectives if available
        if "objectives" in context:
            objectives = context["objectives"]
            if isinstance(objectives, list):
                context_parts.append(f"Objectives: {', '.join(objectives)}")
            else:
                context_parts.append(f"Objectives: {objectives}")
                
        # Add any custom context (limited to avoid token bloat)
        for key, value in context.items():
            if key not in essential_keys and key != "objectives":
                # Skip large data structures to save tokens
                if isinstance(value, (list, dict)) and len(str(value)) > 100:
                    continue
                    
                # Include other relevant context
                context_parts.append(f"{key}: {value}")
                
        return "\n".join(context_parts)
    
    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """
        Estimate token count for a prompt and response.
        
        Args:
            prompt: Prompt text
            response: Response text
            
        Returns:
            Estimated token count
        """
        # Very rough estimate: ~4 chars per token
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4
        return prompt_tokens + response_tokens
    
    def get_token_usage(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Args:
            agent_id: Agent to get stats for (optional, if None returns global stats)
            
        Returns:
            Token usage statistics
        """
        if agent_id:
            return self.agent_usage.get(agent_id, {})
        else:
            return {
                "by_model": self.token_usage,
                "by_task": self.task_usage,
                "by_agent": {agent: sum(usage.values()) for agent, usage in self.agent_usage.items()},
                "total": sum(self.token_usage.values()),
                "fallbacks": self.fallback_counts
            }
    
    def dual_llm_feedback(
        self,
        prompt: str,
        agent_id: Optional[str] = None,
        task_type: str = "tactical"
    ) -> str:
        """
        Get feedback from two LLMs for better quality.
        First from local model, then critiqued by GPT.
        
        Args:
            prompt: The prompt to send
            agent_id: Agent making the request
            task_type: Type of task
            
        Returns:
            Refined response after dual feedback
        """
        # Get initial response from local LLM if available
        local_response = None
        if self.seneca is not None:
            try:
                console.print("[blue]🔄 Getting response from Seneca LLM[/blue]")
                local_response = self.seneca.query(prompt)
            except Exception as e:
                console.print(f"[yellow]⚠ Seneca LLM error: {e}[/yellow]")
                
        # If Seneca failed, try Lily
        if (local_response is None or local_response.strip() == "") and self.lily is not None:
            try:
                console.print("[cyan]🔄 Getting response from Lily LLM[/cyan]")
                lily_prompt = "You are a concise assistant: answer succinctly.\n" + prompt
                local_response = self.lily.query(lily_prompt)
            except Exception as e:
                console.print(f"[yellow]⚠ Lily LLM error: {e}[/yellow]")
                
        # If local response failed, generate initial response with GPT
        if local_response is None or local_response.strip() == "":
            if self.gpt is None:
                return "Error: No LLM models available"
                
            local_response = self.gpt.smart_decision(task_type, prompt)
            
        # Now have GPT critique and refine the local response
        critique_prompt = f"""Review and improve this response to the following request:

Original request: {prompt}

Initial response: {local_response}

Please refine the response to make it more accurate, helpful, and concise. 
If the initial response is already optimal, you may keep it as is.
Respond with ONLY the improved version, no explanations."""
        
        # Get GPT's critique if available, otherwise return local response
        if self.gpt is None:
            return local_response
            
        refined_response = self.gpt.smart_decision("reasoning", critique_prompt)
        return refined_response
    
    def optimize_prompt(self, prompt: str, max_tokens: int = 800) -> str:
        """
        Optimize a prompt to fit within token limit.
        
        Args:
            prompt: Prompt to optimize
            max_tokens: Target maximum tokens
            
        Returns:
            Optimized prompt
        """
        # Estimate current tokens
        current_tokens = len(prompt) // 4
        
        # If already within limit, return as is
        if current_tokens <= max_tokens:
            return prompt
            
        # Simple truncation strategy
        ratio = max_tokens / current_tokens
        char_limit = int(len(prompt) * ratio * 0.9)  # 10% safety margin
        
        # Truncate while preserving structure
        lines = prompt.split('\n')
        result = []
        char_count = 0
        
        for line in lines:
            if char_count + len(line) > char_limit:
                # Add truncation notice
                result.append("... [truncated for token efficiency]")
                break
                
            result.append(line)
            char_count += len(line) + 1  # +1 for newline
            
        return '\n'.join(result)
    
    def clear_cache(self):
        """Clear the in-memory and on-disk cache."""
        self.in_memory_cache = {}
        cache_file = os.path.join(self.cache_dir, "response_cache.json")
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                console.print("[green]✓ Cache cleared[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ Failed to clear cache file: {e}[/yellow]")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive orchestrator usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        stats = {
            "token_usage": self.token_usage,
            "agent_usage": self.agent_usage,
            "task_usage": self.task_usage,
            "fallbacks": self.fallback_counts,
            "cache_size": len(self.in_memory_cache),
            "models_available": {
                "lily": self.lily is not None,
                "seneca": self.seneca is not None,
                "gpt": self.gpt is not None
            }
        }
        
        # Calculate average response times (if tracked)
        if hasattr(self, "response_times"):
            stats["avg_response_times"] = {
                model: sum(times) / len(times) if times else 0
                for model, times in self.response_times.items()
            }
            
        return stats
    
    def display_stats(self):
        """Display rich formatted statistics in the console."""
        stats = self.get_stats()
        
        # Create a table for token usage by model
        token_table = Table(title="Token Usage by Model")
        token_table.add_column("Model", style="cyan")
        token_table.add_column("Tokens", style="green")
        
        for model, tokens in stats["token_usage"].items():
            token_table.add_row(model, f"{tokens:,}")
            
        # Create a table for agent usage
        agent_table = Table(title="Token Usage by Agent")
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Tokens", style="green")
        
        # Fix: Use agent_usage for per-agent stats
        for agent, usage in stats["agent_usage"].items():
            total = sum(usage.values())
            agent_table.add_row(agent, f"{total:,}")
            
        # Create a table for fallbacks
        fallback_table = Table(title="Model Fallbacks")
        fallback_table.add_column("Model", style="cyan")
        fallback_table.add_column("Count", style="yellow")
        
        for model, count in stats["fallbacks"].items():
            fallback_table.add_row(model, str(count))
            
        # Display all tables
        console.print(Panel.fit(f"Total Tokens: {sum(stats['token_usage'].values()):,}"))
        console.print(token_table)
        console.print(agent_table)
        console.print(fallback_table)
        console.print(f"Cache size: {stats['cache_size']} items")

# ─────────────────────────────────────────────
# 🚀 CLI Test Mode
# ─────────────────────────────────────────────
if __name__ == "__main__":
    console.print("[bold magenta]🚀 Testing LLM Orchestrator[/bold magenta]")
    
    llm_orchestrator = LLMOrchestrator()
    
    # Test simple query
    prompt = "What are the top 3 reconnaissance tools used in cybersecurity?"
    console.print(f"[cyan]Prompt:[/cyan] {prompt}")
    
    response = llm_orchestrator.route_task("tactical", prompt, agent_id="TestAgent")
    console.print(f"[green]Response:[/green] {response}")
    
    # Test structured output with scan request
    context = {
        "mission_phase": "reconnaissance",
        "discovered_hosts": ["192.168.1.1", "192.168.1.2"],
        "discovered_services": {"192.168.1.1": ["ssh", "http"]},
        "previous_actions": ["scan 192.168.1.1 -p 1-1000", "fingerprint 192.168.1.1"]
    }
    
    scan_strategy = llm_orchestrator.request_strategy(context, "scan", agent_id="ScoutAgent")
    console.print("[blue]Scan Strategy:[/blue]")
    console.print(scan_strategy)
    
    # Test dual LLM feedback
    dual_response = llm_orchestrator.dual_llm_feedback(prompt, agent_id="OrionAgent")
    console.print(f"[yellow]Dual LLM Response:[/yellow] {dual_response}")
    
    # Show token usage statistics
    llm_orchestrator.display_stats()
