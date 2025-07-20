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
logger = logging.getLogger("ariaska.llm_orchestrator")

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
    - Circular dependency resolution with lazy loading
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
        "gpt-4o": 128000,
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
    
    # Singleton instance for lazy loading to prevent circular imports
    _instance = None
    
    @classmethod
    def get_instance(cls, *args, **kwargs):
        """
        Get or create the singleton instance of LLMOrchestrator.
        This helps resolve circular dependencies by allowing lazy loading.
        
        Returns:
            LLMOrchestrator instance
        """
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance
    
    def __init__(self, 
                 cache_dir: str = "Ariaska_RL/core/memory/llm_cache", 
                 tracking_enabled: bool = True,
                 max_retries: int = 3,
                 lazy_loading: bool = False):
        """
        Initialize the LLM orchestrator.
        
        Args:
            cache_dir: Directory for caching LLM responses
            tracking_enabled: Whether to track token usage
            max_retries: Maximum retries for each model tier
            lazy_loading: Whether to use lazy loading for components (helps resolve circular imports)
        """
        self.cache_dir = cache_dir
        self.tracking_enabled = tracking_enabled
        self.max_retries = max_retries
        self.in_memory_cache = {}
        self.token_usage = {}
        self.agent_usage = {}
        self.task_usage = {}
        self.fallback_counts = {}
        self.response_times = {}
        self._lazy_loading = lazy_loading
        self._gpt_manager = None  # For lazy loading GPTManager
        
        # Ensure cache directory exists - create full path
        try:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_available = True
        except Exception as e:
            logger.error(f"Failed to create cache directory {cache_dir}: {e}")
            console.print(f"[yellow]⚠ Failed to create cache directory: {e}[/yellow]")
            self.cache_available = False
        
        # Initialize local models
        try:
            self.lily = LocalLLMManager(model_name=os.environ.get(
                "ARIASKA_LILY_MODEL", "QuantFactory/Lily-Cybersecurity-7B-v0.2-GGUF:Q8_0"))
            console.print("[green]✓ Lily LLM initialized[/green]")
        except Exception as e:
            logger.warning(f"Could not initialize Lily LLM: {e}")
            console.print(f"[yellow]⚠ Could not initialize Lily LLM: {e}[/yellow]")
            self.lily = None
            
        try:
            self.seneca = LocalLLMManager(model_name=os.environ.get(
                "ARIASKA_SENECA_MODEL", "wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF"))
            console.print("[green]✓ Seneca LLM initialized[/green]")
        except Exception as e:
            logger.warning(f"Could not initialize Seneca LLM: {e}")
            console.print(f"[yellow]⚠ Could not initialize Seneca LLM: {e}[/yellow]")
            self.seneca = None
        
        # Initialize GPT Manager (unless lazy loading is enabled)
        if not self._lazy_loading:
            try:
                from core.gpt_manager import GPTManager
                self.gpt = GPTManager()
                console.print("[green]✓ GPT Manager initialized[/green]")
            except Exception as e:
                logger.error(f"Could not initialize GPT Manager: {e}")
                console.print(f"[yellow]⚠ Could not initialize GPT Manager: {e}[/yellow]")
                self.gpt = None
        else:
            self.gpt = None
            
        # Load cache from disk if available
        if self.cache_available:
            self._load_cache()
        
        console.print(f"[green]✓ LLM Orchestrator initialized (cache: {len(self.in_memory_cache)} items)[/green]")
    
    @property
    def gpt_manager(self):
        """
        Lazy loading property for GPTManager.
        Only imports and initializes when first accessed to avoid circular imports.
        
        Returns:
            GPTManager instance
        """
        if self._gpt_manager is None and self._lazy_loading:
            try:
                from core.gpt_manager import GPTManager
                self._gpt_manager = GPTManager()
                console.print("[green]✓ GPT Manager lazy-loaded[/green]")
            except Exception as e:
                logger.error(f"Could not lazy-load GPT Manager: {e}")
                console.print(f"[yellow]⚠ Could not lazy-load GPT Manager: {e}[/yellow]")
                self._gpt_manager = None
        return self._gpt_manager or self.gpt  # Return either the lazy-loaded manager or the one from init
        
    def _load_cache(self):
        """Load LLM response cache from disk."""
        cache_file = os.path.join(self.cache_dir, "response_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self.in_memory_cache = json.load(f)
                console.print(f"[green]✓ Loaded {len(self.in_memory_cache)} cached responses[/green]")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                console.print(f"[yellow]⚠ Failed to load cache: {e}[/yellow]")
                self.in_memory_cache = {}
                
    def _save_cache(self):
        """Save LLM response cache to disk."""
        if not self.cache_available:
            return
            
        cache_file = os.path.join(self.cache_dir, "response_cache.json")
        try:
            with open(cache_file, "w") as f:
                json.dump(self.in_memory_cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
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
        if use_cache and self.cache_available:
            cache_key = self._get_cache_key(prompt, task_type)
            if cache_key in self.in_memory_cache:
                return self.in_memory_cache[cache_key]
        
        t0 = time.time()
        
        # Safety check - ensure prompt isn't empty
        if not prompt or len(prompt.strip()) < 5:
            error_msg = "Empty or too short prompt provided"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
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
            if model_name == "gpt" and self.gpt is None and self.gpt_manager is None:
                continue
                
            # Track response times for this model
            self.response_times.setdefault(model_name, [])
            model_start_time = time.time()
                
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
                        # Use either directly initialized gpt or lazily loaded gpt_manager
                        gpt_instance = self.gpt_manager if self._lazy_loading else self.gpt
                        
                        if task_type == "dual-llm-feedback":
                            raw_response = gpt_instance.dual_llm_feedback(prompt)
                        else:
                            # Use smart_decision if available, fallback to gpt_request
                            if hasattr(gpt_instance, "smart_decision"):
                                raw_response = gpt_instance.smart_decision(task_type, prompt)
                            else:
                                raw_response = gpt_instance.gpt_request(prompt, agent_id=agent_id)
                        model_used = "GPT-4o-mini"
                    
                    # Record response time
                    model_end_time = time.time()
                    self.response_times[model_name].append(model_end_time - model_start_time)
                    
                    # Validate response if schema provided
                    if schema is not None:
                        validated_response = self._validate_and_fix_output(raw_response, schema, model_name, prompt)
                        if validated_response:
                            response = validated_response
                            break
                    else:
                        # Basic validation - not empty, not too long, not gibberish
                        if raw_response and len(raw_response) < 2000 and raw_response.count(" ") >= 2:
                            response = raw_response
                            break
                        else:
                            # Invalid response, log and continue
                            logger.warning(f"Invalid response from {model_name}: empty or too short")
                        
                except Exception as e:
                    last_error = e
                    logger.error(f"{model_name} error (retry {retry+1}/{self.max_retries}): {str(e)}")
                    console.print(f"[yellow]⚠️ {model_name} error (retry {retry+1}/{self.max_retries}): {str(e)}[/yellow]")
                    
                    # Wait a bit before retrying
                    time.sleep(min(0.5 * (retry + 1), 2.0))
                    
                # If we get here, this model tier failed
                if retry == self.max_retries - 1:
                    logger.error(f"{model_name} failed after {self.max_retries} retries")
                    console.print(f"[red]❌ {model_name} failed after {self.max_retries} retries[/red]")
                    # Track fallbacks
                    self.fallback_counts.setdefault(model_name, 0)
                    self.fallback_counts[model_name] += 1
        
        # If all models failed
        if response is None:
            error_msg = f"All LLM models failed. Last error: {last_error}"
            logger.critical(error_msg)
            console.print(f"[red]{error_msg}[/red]")
            
            # Emergency response - just provide a basic message
            if task_type == "strategic":
                return "Recommend proceeding with caution and gathering more information."
            elif task_type == "tactical":
                return "Execute basic scans first to discover system information."
            else:
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
        if use_cache and self.cache_available:
            cache_key = self._get_cache_key(prompt, task_type)
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
                logger.error(f"Failed to validate response: {e}")
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
                    if hasattr(self.gpt, "smart_decision"):
                        fixed_response = self.gpt.smart_decision("fix", fix_prompt)
                    else:
                        fixed_response = self.gpt.gpt_request(fix_prompt, agent_id="system")
                    
                    # Verify the fixed response
                    json_data = self._extract_json_from_text(fixed_response)
                    schema(**json_data)  # This will raise ValidationError if still invalid
                    return fixed_response
            except Exception as e:
                logger.error(f"Failed to fix output: {e}")
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
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If parsing failed, try to clean the string
                json_str = re.sub(r'```json|```', '', json_str).strip()
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
                logger.warning(f"Seneca LLM error: {e}")
                console.print(f"[yellow]⚠ Seneca LLM error: {e}[/yellow]")
                
        # If Seneca failed, try Lily
        if (local_response is None or local_response.strip() == "") and self.lily is not None:
            try:
                console.print("[cyan]🔄 Getting response from Lily LLM[/cyan]")
                lily_prompt = "You are a concise assistant: answer succinctly.\n" + prompt
                local_response = self.lily.query(lily_prompt)
            except Exception as e:
                logger.warning(f"Lily LLM error: {e}")
                console.print(f"[yellow]⚠ Lily LLM error: {e}[/yellow]")
                
        # If local response failed, generate initial response with GPT
        if local_response is None or local_response.strip() == "":
            if self.gpt is None:
                logger.error("No LLM models available")
                return "Error: No LLM models available"
                
            try:
                if hasattr(self.gpt, "smart_decision"):
                    local_response = self.gpt.smart_decision(task_type, prompt)
                else:
                    local_response = self.gpt.gpt_request(prompt, agent_id=agent_id)
            except Exception as e:
                logger.error(f"GPT error: {e}")
                return "Error: All LLM models failed"
            
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
            
        try:
            if hasattr(self.gpt, "smart_decision"):
                refined_response = self.gpt.smart_decision("reasoning", critique_prompt)
            else:
                refined_response = self.gpt.gpt_request(critique_prompt, agent_id=agent_id)
            return refined_response
        except Exception as e:
            logger.error(f"GPT critique error: {e}")
            return local_response
    
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
        if not self.cache_available:
            return
            
        cache_file = os.path.join(self.cache_dir, "response_cache.json")
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                console.print("[green]✓ Cache cleared[/green]")
            except Exception as e:
                logger.error(f"Failed to clear cache file: {e}")
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

    def health_check(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Run a comprehensive health check on all LLM components.
        
        Args:
            detailed: Whether to include detailed model responses in results
            
        Returns:
            Dict with health status information
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "status": "unknown",
            "latency_tests": {}
        }
        
        # Check local LLMs
        results["components"]["lily"] = self._check_local_llm(self.lily, "lily")
        results["components"]["seneca"] = self._check_local_llm(self.seneca, "seneca")
        
        # Check GPT connection
        gpt_instance = self.gpt_manager if self._lazy_loading else self.gpt
        results["components"]["gpt"] = self._check_gpt(gpt_instance)
        
        # Check cache status
        results["components"]["cache"] = self._check_cache()
        
        # Run latency tests if requested
        if detailed:
            results["latency_tests"] = self._run_latency_tests()
            
        # Determine overall status
        status_values = [comp.get("status") for comp in results["components"].values()]
        
        if all(status == "operational" for status in status_values):
            results["status"] = "operational"
        elif all(status in ("operational", "degraded", "not_available") for status in status_values):
            results["status"] = "degraded"
        else:
            results["status"] = "critical"
            
        return results
        
    def _check_local_llm(self, llm, name: str) -> Dict[str, Any]:
        """Check health of a local LLM."""
        if llm is None:
            return {
                "status": "not_available",
                "error": f"{name} not initialized"
            }
            
        try:
            # Try a simple query to test responsiveness
            t0 = time.time()
            response = llm.query("Give a one word response for a test.")
            latency = time.time() - t0
            
            # Check if response is reasonable
            if response and len(response.strip()) > 0:
                return {
                    "status": "operational",
                    "latency": latency,
                    "model_name": getattr(llm, "model_name", "unknown")
                }
            else:
                return {
                    "status": "degraded",
                    "error": "Empty or invalid response",
                    "latency": latency
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
            
    def _check_gpt(self, gpt_instance) -> Dict[str, Any]:
        """Check health of GPT service."""
        if gpt_instance is None:
            return {
                "status": "not_available",
                "error": "GPT not initialized"
            }
            
        try:
            # Try a simple API call to test connectivity
            t0 = time.time()
            
            # Use different methods based on what's available
            if hasattr(gpt_instance, "gpt_test_connection"):
                response = gpt_instance.gpt_test_connection()
            elif hasattr(gpt_instance, "gpt_request"):
                response = gpt_instance.gpt_request("Respond with 'ok' for a connection test.")
            else:
                return {
                    "status": "unknown",
                    "error": "No suitable test method found"
                }
                
            latency = time.time() - t0
            
            # Check if we got a valid response
            if response and isinstance(response, str) and len(response.strip()) > 0:
                return {
                    "status": "operational",
                    "latency": latency,
                    "default_model": getattr(gpt_instance, "default_model", "unknown")
                }
            else:
                return {
                    "status": "degraded",
                    "error": "Empty or invalid response",
                    "latency": latency
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
            
    def _check_cache(self) -> Dict[str, Any]:
        """Check LLM cache health."""
        result = {
            "status": "operational",
            "in_memory_size": len(self.in_memory_cache),
            "disk_available": self.cache_available
        }
        
        # Check if disk cache is accessible
        if self.cache_available:
            try:
                cache_files = os.listdir(self.cache_dir)
                result["disk_cache_files"] = len(cache_files)
            except Exception as e:
                result["status"] = "degraded"
                result["error"] = f"Disk cache error: {str(e)}"
                
        return result
        
    def _run_latency_tests(self) -> Dict[str, Any]:
        """Run latency tests on all available models."""
        results = {}
        test_prompt = "Respond with a single word for testing purposes."
        
        # Test Lily (Tier 1)
        if self.lily is not None:
            try:
                t0 = time.time()
                response = self.lily.query(test_prompt)
                latency = time.time() - t0
                
                results["tier1_local_small"] = {
                    "status": "success",
                    "model": "lily",
                    "latency": latency,
                    "response": response[:20] + "..." if len(response) > 20 else response
                }
            except Exception as e:
                results["tier1_local_small"] = {
                    "status": "error",
                    "model": "lily",
                    "error": str(e)
                }
                
        # Test Seneca (Tier 2)
        if self.seneca is not None:
            try:
                t0 = time.time()
                response = self.seneca.query(test_prompt)
                latency = time.time() - t0
                
                results["tier2_local_medium"] = {
                    "status": "success", 
                    "model": "seneca",
                    "latency": latency,
                    "response": response[:20] + "..." if len(response) > 20 else response
                }
            except Exception as e:
                results["tier2_local_medium"] = {
                    "status": "error",
                    "model": "seneca",
                    "error": str(e)
                }
                
        # Test GPT (Tier 4)
        gpt_instance = self.gpt_manager if self._lazy_loading else self.gpt
        if gpt_instance is not None:
            try:
                t0 = time.time()
                
                if hasattr(gpt_instance, "gpt_request"):
                    response = gpt_instance.gpt_request(test_prompt)
                    
                    results["tier4_cloud_large"] = {
                        "status": "success",
                        "model": getattr(gpt_instance, "default_model", "gpt"),
                        "latency": time.time() - t0,
                        "response": response[:20] + "..." if len(response) > 20 else response
                    }
                else:
                    results["tier4_cloud_large"] = {
                        "status": "error",
                        "model": "gpt",
                        "error": "No gpt_request method available"
                    }
            except Exception as e:
                results["tier4_cloud_large"] = {
                    "status": "error",
                    "model": "gpt",
                    "error": str(e)
                }
                
        return results
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed usage statistics for all LLM operations.
        
        Returns:
            Dict with comprehensive usage statistics
        """
        stats = {
            "timestamp": datetime.now().isoformat(),
            "token_usage": {
                "total": sum(self.token_usage.values()) if self.token_usage else 0,
                "by_model": self.token_usage.copy() if self.token_usage else {},
                "by_agent": self.agent_usage.copy() if self.agent_usage else {},
                "by_task": self.task_usage.copy() if self.task_usage else {}
            },
            "cache": {
                "size": len(self.in_memory_cache),
                "hit_rate": 0.0  # Default value
            },
            "fallbacks": {
                "total": sum(self.fallback_counts.values()) if self.fallback_counts else 0,
                "by_model": self.fallback_counts.copy() if self.fallback_counts else {}
            },
            "latency": {
                "average_by_model": {}
            },
            "top_prompts": []
        }
        
        # Calculate hit rate if tracking is enabled
        if hasattr(self, "cache_hits") and hasattr(self, "cache_misses"):
            total_queries = self.cache_hits + self.cache_misses
            stats["cache"]["hit_rate"] = self.cache_hits / total_queries if total_queries > 0 else 0.0
            stats["cache"]["hits"] = self.cache_hits
            stats["cache"]["misses"] = self.cache_misses
            
        # Calculate average latency by model
        for model_name, times in self.response_times.items():
            if times:
                stats["latency"]["average_by_model"][model_name] = sum(times) / len(times)
                
        # Get most frequent prompts (anonymized for privacy)
        if hasattr(self, "prompt_frequency") and self.prompt_frequency:
            prompt_items = sorted(
                self.prompt_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5
            
            for prompt, count in prompt_items:
                # Safely truncate and anonymize prompt
                safe_prompt = prompt[:30] + "..." if len(prompt) > 30 else prompt
                stats["top_prompts"].append({
                    "prompt": safe_prompt,
                    "count": count
                })
                
        return stats
        
    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear the LLM response cache (both in-memory and on disk).
        
        Returns:
            Dict with results of the operation
        """
        result = {
            "in_memory_cleared": len(self.in_memory_cache),
            "disk_cleared": 0,
            "errors": []
        }
        
        # Clear in-memory cache
        self.in_memory_cache.clear()
        
        # Clear disk cache if available
        if self.cache_available:
            try:
                cache_files = os.listdir(self.cache_dir)
                for file in cache_files:
                    if file.endswith(".json"):
                        try:
                            os.remove(os.path.join(self.cache_dir, file))
                            result["disk_cleared"] += 1
                        except Exception as e:
                            result["errors"].append(f"Failed to remove {file}: {e}")
            except Exception as e:
                result["errors"].append(f"Failed to access cache directory: {e}")
                
        # Reset cache counters if they exist
        if hasattr(self, "cache_hits"):
            self.cache_hits = 0
        if hasattr(self, "cache_misses"):
            self.cache_misses = 0
            
        return result
        
    def display_stats(self) -> None:
        """
        Display usage statistics in a user-friendly format.
        """
        stats = self.get_stats()
        
        try:
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            
            # Token usage table
            token_table = Table(title="LLM Token Usage")
            token_table.add_column("Category")
            token_table.add_column("Count", justify="right")
            
            token_table.add_row("Total", f"{stats['token_usage']['total']:,}")
            
            # By model
            for model, count in stats['token_usage']['by_model'].items():
                token_table.add_row(f"Model: {model}", f"{count:,}")
                
            # By agent (top 3)
            agent_items = sorted(
                stats['token_usage']['by_agent'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            for agent, count in agent_items:
                token_table.add_row(f"Agent: {agent}", f"{count:,}")
                
            console.print(token_table)
            
            # Performance table
            perf_table = Table(title="LLM Performance")
            perf_table.add_column("Metric")
            perf_table.add_column("Value")
            
            # Cache stats
            perf_table.add_row("Cache Size", f"{stats['cache']['size']} entries")
            perf_table.add_row("Cache Hit Rate", f"{stats['cache']['hit_rate']:.1%}")
            
            # Latency
            for model, latency in stats['latency']['average_by_model'].items():
                perf_table.add_row(f"{model} Latency", f"{latency:.2f}s")
                
            # Fallbacks
            perf_table.add_row("Total Fallbacks", f"{stats['fallbacks']['total']}")
            
            console.print(perf_table)
            
        except ImportError:
            # Fallback to basic print
            print("LLM Token Usage:")
            print(f"Total: {stats['token_usage']['total']:,}")
            
            print("\nBy Model:")
            for model, count in stats['token_usage']['by_model'].items():
                print(f"  {model}: {count:,}")
                
            print("\nPerformance:")
            print(f"Cache Size: {stats['cache']['size']} entries")
            print(f"Cache Hit Rate: {stats['cache']['hit_rate']:.1%}")
            
            for model, latency in stats['latency']['average_by_model'].items():
                print(f"{model} Latency: {latency:.2f}s")
                
            print(f"Total Fallbacks: {stats['fallbacks']['total']}")
    
    def _track_tokens(self, agent_id: Optional[str], task_type: str, model: str, token_count: int) -> None:
        """
        Track token usage by agent, task, and model.
        
        Args:
            agent_id: Agent ID using the LLM
            task_type: Type of task
            model: Model used
            token_count: Number of tokens used
        """
        if not self.tracking_enabled:
            return
            
        # Track by model
        self.token_usage.setdefault(model, 0)
        self.token_usage[model] += token_count
        
        # Track by agent
        if agent_id:
            self.agent_usage.setdefault(agent_id, 0)
            self.agent_usage[agent_id] += token_count
            
        # Track by task
        if task_type:
            self.task_usage.setdefault(task_type, 0)
            self.task_usage[task_type] += token_count
            
        # Track cache performance
        if not hasattr(self, "cache_hits"):
            self.cache_hits = 0
            self.cache_misses = 0
            
        # Track prompt frequency (for analysis)
        if not hasattr(self, "prompt_frequency"):
            self.prompt_frequency = {}
            
    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """
        Estimate token count for prompt and response.
        
        Args:
            prompt: Input prompt
            response: Model response
            
        Returns:
            Estimated token count
        """
        # Simple estimation based on whitespace-separated words
        # About 4 characters per token on average for English text
        prompt_tokens = len(prompt) / 4
        response_tokens = len(response) / 4
        
        return int(prompt_tokens + response_tokens)
            
    def _get_cache_key(self, prompt: str, task_type: str = None) -> str:
        """
        Generate a cache key for a prompt.
        
        Args:
            prompt: The prompt to hash
            task_type: Optional task type to include in the key
            
        Returns:
            Cache key string
        """
        # Create a hash of the prompt to use as key
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        if task_type:
            return f"{task_type}_{prompt_hash}"
        return prompt_hash
        
    def _load_cache(self) -> None:
        """Load cached responses from disk."""
        try:
            # Find all cache files
            cache_files = glob.glob(os.path.join(self.cache_dir, "*.json"))
            
            for file in cache_files:
                try:
                    with open(file, 'r') as f:
                        cache_data = json.load(f)
                        self.in_memory_cache.update(cache_data)
                except Exception as e:
                    logger.error(f"Error loading cache file {file}: {e}")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            
    def _save_cache(self) -> None:
        """Save cached responses to disk."""
        if not self.cache_available:
            return
            
        try:
            # Use a timestamp to ensure unique filenames
            timestamp = int(time.time())
            cache_file = os.path.join(self.cache_dir, f"cache_{timestamp}.json")
            
            # Don't write empty cache
            if not self.in_memory_cache:
                return
                
            with open(cache_file, 'w') as f:
                json.dump(self.in_memory_cache, f)
                
            # Manage cache file count - keep only the 5 most recent
            try:
                cache_files = sorted(
                    glob.glob(os.path.join(self.cache_dir, "cache_*.json")), 
                    reverse=True
                )
                
                for old_file in cache_files[5:]:
                    os.remove(old_file)
            except Exception as e:
                logger.error(f"Error managing cache files: {e}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            
    def _validate_and_fix_output(self, output: str, schema: BaseModel, model_name: str, original_prompt: str) -> Optional[str]:
        """
        Validate output against a Pydantic schema and attempt to fix if invalid.
        
        Args:
            output: Raw LLM output
            schema: Pydantic schema to validate against
            model_name: Name of the model that generated the output
            original_prompt: The original prompt sent to the model
            
        Returns:
            Validated output or None if validation failed
        """
        # First try to validate the raw output
        try:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                # If output is JSON, parse and validate
                if output.strip().startswith("{") and output.strip().endswith("}"):
                    data = json.loads(output)
                    validated = schema(**data)
                    return output
                    
                # Try to extract JSON from text
                json_match = re.search(r"```json\n(.*?)\n```", output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    data = json.loads(json_str)
                    validated = schema(**data)
                    return json_str
                    
                # Try to extract using more lenient pattern
                json_data = self._extract_json_from_text(output)
                if json_data:
                    validated = schema(**json_data)
                    return json.dumps(json_data)
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            
        # If we reach here, validation failed - attempt to fix with GPT
        try:
            gpt_instance = self.gpt_manager if self._lazy_loading else self.gpt
            if gpt_instance is None:
                return None
                
            fix_prompt = f"""The following output needs to be fixed to match this JSON schema:
{schema.schema_json()}

Original output:
{output}

Please provide ONLY a valid JSON object matching the schema, nothing else."""

            fixed_output = gpt_instance.gpt_request(fix_prompt)
            
            # Try to validate the fixed output
            json_data = self._extract_json_from_text(fixed_output)
            if json_data:
                validated = schema(**json_data)
                return json.dumps(json_data)
        except Exception as e:
            logger.error(f"Fix attempt failed: {e}")
            
        return None
        
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from text that might contain markdown, code blocks, etc.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Parsed JSON object or None if no valid JSON found
        """
        # Try various patterns to extract JSON
        patterns = [
            # Code block with json
            r"```json\n(.*?)\n```",
            # Code block without language specifier
            r"```\n(.*?)\n```", 
            # Just curly braces
            r"(\{.*\})"
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, text, re.DOTALL)
            if matches:
                try:
                    json_str = matches.group(1)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                    
        # Final attempt - try to find anything that looks like JSON
        try:
            # Look for content between outermost braces
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
            
        return None
        
    def optimize_prompt(self, prompt: str, model: str, max_tokens: Optional[int] = None) -> str:
        """
        Optimize a prompt to fit within token constraints.
        
        Args:
            prompt: The original prompt
            model: Target model name
            max_tokens: Maximum tokens (if None, uses model default)
            
        Returns:
            Optimized prompt
        """
        if max_tokens is None:
            max_tokens = self.TOKEN_LIMITS.get(model, 4000)
            
        # If prompt is already well below limit, return as is
        estimated_tokens = len(prompt) / 4  # Rough estimate
        if estimated_tokens < max_tokens * 0.9:
            return prompt
            
        # Split into sections for smarter trimming
        sections = re.split(r'(\n\n|\n#+\s)', prompt)
        
        # If prompt has context block, preprocess it specially
        if "Context:" in prompt:
            return self._preprocess_context(prompt, max_tokens)
            
        # Otherwise perform general trimming
        result = []
        current_length = 0
        target_length = max_tokens * 4  # Convert back to chars for simplicity
        
        # Always keep the first section (instructions)
        result.append(sections[0])
        current_length += len(sections[0])
        
        # Process remaining sections
        for i in range(1, len(sections)):
            section = sections[i]
            section_length = len(section)
            
            # If adding this section would exceed target, skip
            if current_length + section_length > target_length:
                # If we've processed less than 50% of sections, do aggressive trimming
                if i < len(sections) / 2:
                    # Trim the section to fit
                    available_space = target_length - current_length
                    if available_space > 100:  # Only if we have reasonable space
                        trimmed = section[:available_space-20] + "..."
                        result.append(trimmed)
                        current_length += len(trimmed)
                        break
                else:
                    # We've already processed most important sections, so we can stop
                    break
            else:
                # Add section as-is
                result.append(section)
                current_length += section_length
                
        return ''.join(result)
        
    def _preprocess_context(self, prompt: str, max_tokens: int) -> str:
        """Optimize a prompt that has a specific context section."""
        # Split into parts before and after Context:
        parts = prompt.split("Context:", 1)
        if len(parts) != 2:
            # If no "Context:" found, use regular optimization
            return self.optimize_prompt(prompt, max_tokens)
            
        instruction_part = parts[0] + "Context:"
        context_part = parts[1]
        
        # Calculate target length
        instruction_tokens = len(instruction_part) / 4
        available_context_tokens = max_tokens - instruction_tokens - 50  # Some buffer
        
        # If context fits, return as-is
        context_tokens = len(context_part) / 4
        if context_tokens <= available_context_tokens:
            return prompt
            
        # Otherwise, trim context
        available_chars = int(available_context_tokens * 4)
        trimmed_context = context_part[:available_chars] + "...[context truncated due to length]"
        
        return instruction_part + trimmed_context

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
    
    # Run a health check
    health = llm_orchestrator.health_check()
    console.print("[bold]🩺 Health Check:[/bold]")
    console.print(health)
    
    # Show token usage statistics
    llm_orchestrator.display_stats()
