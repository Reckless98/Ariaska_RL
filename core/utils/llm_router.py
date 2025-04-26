# core/utils/llm_router.py — ARIASKA LLM Router v1.0
# Centralized LLM orchestration system with fallback chain, validation, and optimization

import os
import time
import json
import hashlib
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Type
from enum import Enum
from pydantic import BaseModel, Field, ValidationError, create_model
import re
import threading
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Core LLM handlers
from core.utils.local_llm_manager import LocalLLMManager
from core.gpt_manager import GPTManager
from core.utils.context_encoder import ContextEncoder

console = Console()
logger = logging.getLogger(__name__)

# ===== Schema Definitions for Command Validation =====

class CommandBase(BaseModel):
    """Base model for all command outputs"""
    command: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reasoning: Optional[str] = None

class ReconCommand(CommandBase):
    """Schema for reconnaissance commands"""
    targets: Optional[List[str]] = None
    ports: Optional[List[int]] = None 
    stealth_level: Optional[int] = Field(default=1, ge=1, le=5)

class ExploitCommand(CommandBase):
    """Schema for exploit commands"""
    target: str
    service: str
    exploit_type: str
    payload: Optional[str] = None
    risk_level: Optional[int] = Field(default=2, ge=1, le=5)

class PrivilegeEscalationCommand(CommandBase):
    """Schema for privilege escalation commands"""
    technique: str 
    target_privilege: str = "root"

class ExfiltrateCommand(CommandBase):
    """Schema for data exfiltration commands"""
    data_type: str
    destination: str
    encryption: Optional[bool] = True

# ===== LLM Request & Response Models =====

class LLMModelType(Enum):
    """Types of LLM models in the fallback chain"""
    LOCAL_SMALL = "lily"       # Lily - Fast, lightweight
    LOCAL_MEDIUM = "seneca"    # Seneca - More capable local
    CLOUD_SMALL = "gpt-nano"   # GPT-4.1-nano - Small cloud model
    CLOUD_LARGE = "gpt-main"   # GPT-4o-mini - Main capable model

class LLMRequest:
    """Container for an LLM request with context and parameters"""
    def __init__(
        self,
        prompt: str,
        role: str = "tactical",
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        schema: Optional[Type[BaseModel]] = None,
        require_validation: bool = True,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        model_preference: Optional[LLMModelType] = None
    ):
        self.prompt = prompt
        self.role = role
        self.agent_id = agent_id
        self.context = context or {}
        self.schema = schema
        self.require_validation = require_validation
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_preference = model_preference
        self.created_at = time.time()

class LLMResponse:
    """Container for an LLM response with metadata"""
    def __init__(
        self,
        content: str,
        model_used: str,
        tokens_used: int,
        elapsed_time: float,
        validated: bool = False,
        parsed: Optional[Dict[str, Any]] = None,
        fallbacks_used: int = 0
    ):
        self.content = content
        self.model_used = model_used
        self.tokens_used = tokens_used
        self.elapsed_time = elapsed_time
        self.validated = validated
        self.parsed = parsed
        self.fallbacks_used = fallbacks_used
        
    def __str__(self) -> str:
        return self.content
        
    def __repr__(self) -> str:
        return f"LLMResponse(model={self.model_used}, tokens={self.tokens_used}, validated={self.validated})"

# ===== Prompt Templates =====

# Structured by role and optimized for token efficiency
PROMPT_TEMPLATES = {
    "tactical": """Using only the essential context, suggest a specific tactical command. 
Context: {context}

Provide ONLY the command itself with no explanation or commentary.""",

    "recon": """Based on target information, suggest a reconnaissance command.
Context: {context}

Output JSON with this structure:
{
  "command": "nmap -sV ...",
  "targets": ["target1", "target2"],
  "ports": [22, 80, 443],
  "stealth_level": 2,
  "confidence": 0.9
}""",

    "exploit": """Based on discovered vulnerabilities, suggest an exploit command.
Context: {context}

Output JSON with this structure:
{
  "command": "exploit_command",
  "target": "target_ip_or_service",
  "service": "service_name",
  "exploit_type": "exploit_name",
  "payload": "optional_payload",
  "risk_level": 1-5,
  "confidence": 0.1-1.0
}""",

    "privesc": """Based on current access level, suggest a privilege escalation command.
Context: {context}

Output JSON with this structure:
{
  "command": "privesc_command",
  "technique": "technique_name",
  "target_privilege": "desired_level",
  "confidence": 0.1-1.0
}""",

    "strategic": """Analyze current mission state and suggest high-level strategy.
Context: {context}

Respond with a concise strategy in 3 bullet points."""
}

# ===== Main LLM Router Class =====

class LLMRouter:
    """
    Centralized LLM orchestration system with:
    - Robust fallback chain: Seneca → Lily → GPT
    - Token-efficient prompt engineering 
    - Command validation with Pydantic schemas
    - Performance optimizations (caching, context windowing)
    """
    
    def __init__(
        self, 
        cache_dir: str = "core/memories/llm_cache",
        max_cache_size: int = 10000,
        token_budget: int = 50000,  # Daily token budget
        default_timeout: int = 10,
        max_retries: int = 3
    ):
        """
        Initialize the LLM Router with specified configuration.
        
        Args:
            cache_dir: Directory to store persistent cache
            max_cache_size: Maximum number of items in memory cache
            token_budget: Daily token budget for cloud models
            default_timeout: Default timeout for LLM requests in seconds
            max_retries: Maximum number of retries for failed requests
        """
        # Basic configuration
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.token_budget = token_budget
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize cache and locks
        self.in_memory_cache = {}
        self.cache_lock = threading.Lock()
        
        # Initialize LLM managers
        try:
            self.lily_llm = LocalLLMManager(model_name="QuantFactory/Lily-Cybersecurity-7B-v0.2-GGUF:Q8_0")
            console.print("[green]✓ Lily LLM initialized[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not initialize Lily LLM: {e}[/yellow]")
            self.lily_llm = None
            
        try:
            self.seneca_llm = LocalLLMManager(model_name="wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF")
            console.print("[green]✓ Seneca LLM initialized[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not initialize Seneca LLM: {e}[/yellow]")
            self.seneca_llm = None
        
        try:
            self.gpt_manager = GPTManager()
            console.print("[green]✓ GPT Manager initialized[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not initialize GPT Manager: {e}[/yellow]")
            self.gpt_manager = None
            
        # Initialize token tracking
        self.token_usage = {
            "lily": 0,
            "seneca": 0,
            "gpt-nano": 0,
            "gpt-main": 0,
            "total": 0
        }
        
        self.agent_usage = {}  # agent_id -> token count
        self.role_usage = {}   # role -> token count
        
        # Performance metrics
        self.request_count = 0
        self.fallback_count = 0
        self.validation_failures = 0
        self.cache_hits = 0
        self.response_times = []
        
        # Schema mapping
        self.schema_map = {
            "recon": ReconCommand,
            "exploit": ExploitCommand,
            "privesc": PrivilegeEscalationCommand,
            "exfiltrate": ExfiltrateCommand
        }
        
        # Load cache
        self._load_cache()
        
        console.print(f"[green]✓ LLM Router initialized with {len(self.in_memory_cache)} cached items[/green]")
        
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
            # Ensure the directory exists
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            with open(cache_file, "w") as f:
                # Only store a limited number of items to prevent cache explosion
                # Sort by most recent first to keep the most relevant items
                sorted_cache = dict(sorted(
                    self.in_memory_cache.items(), 
                    key=lambda x: x[1].get('timestamp', 0), 
                    reverse=True
                )[:self.max_cache_size])
                
                json.dump(sorted_cache, f)
                
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to save cache: {e}[/yellow]")
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """
        Generate a unique cache key for a request.
        
        Args:
            request: The LLM request
            
        Returns:
            A unique cache key string
        """
        # Include role, prompt, and essential context elements in the key
        key_parts = [
            request.role,
            request.prompt.strip()[:150]  # Truncate long prompts
        ]
        
        # Add critical context elements that would affect the response
        critical_context_keys = ['targets', 'hosts', 'ports', 'phase', 'privilege_level']
        if request.context:
            for key in critical_context_keys:
                if key in request.context:
                    context_value = str(request.context[key])[:50]  # Truncate long values
                    key_parts.append(f"{key}:{context_value}")
                    
        # Create a hash of the combined key parts
        combined = "|".join(key_parts)
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Simple approximation: ~4 characters per token
        return max(1, len(text) // 4)
    
    def _optimize_prompt(self, request: LLMRequest) -> str:
        """
        Optimize a prompt by applying the appropriate template and minimizing context.
        
        Args:
            request: The LLM request
            
        Returns:
            Optimized prompt string
        """
        # Get the appropriate template
        template = PROMPT_TEMPLATES.get(request.role, PROMPT_TEMPLATES["tactical"])
        
        # Format the context if available
        context_str = "No context provided"
        if request.context:
            # Use ContextEncoder to create a concise summary if possible
            if hasattr(request.context, 'get') and request.context.get('state'):
                try:
                    context_str = ContextEncoder.summarize_agent_state(request.context['state'])
                except Exception:
                    # Fall back to manual context formatting
                    context_parts = []
                    for k, v in request.context.items():
                        # Skip large values and non-essential keys
                        if (isinstance(v, (list, dict)) and len(str(v)) > 100) or k in ['history', 'log']:
                            continue
                        context_parts.append(f"{k}: {v}")
                    context_str = " | ".join(context_parts)
            else:
                # Manual context formatting
                context_parts = []
                for k, v in request.context.items():
                    if (isinstance(v, (list, dict)) and len(str(v)) > 100):
                        continue
                    context_parts.append(f"{k}: {v}")
                context_str = " | ".join(context_parts)
        
        # Format the template with the prompt and context
        optimized = template.format(
            context=context_str,
            prompt=request.prompt
        )
        
        # Ensure prompt fits within token budget if specified
        if request.max_tokens:
            current_tokens = self._estimate_tokens(optimized)
            if current_tokens > request.max_tokens:
                # Scale down by keeping the template structure but truncating context
                scaling_ratio = request.max_tokens / current_tokens * 0.9  # 10% safety margin
                max_context_chars = int(len(context_str) * scaling_ratio)
                truncated_context = context_str[:max_context_chars] + "..."
                optimized = template.format(
                    context=truncated_context,
                    prompt=request.prompt
                )
        
        return optimized
    
    def _get_schema_for_role(self, role: str) -> Optional[Type[BaseModel]]:
        """
        Get the appropriate Pydantic schema for validation based on role.
        
        Args:
            role: The role type (recon, exploit, etc.)
            
        Returns:
            Pydantic schema class or None
        """
        return self.schema_map.get(role)
        
    def _parse_json_from_output(self, output: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from LLM output text.
        
        Args:
            output: Raw output text from LLM
            
        Returns:
            Extracted JSON as dict or None if extraction fails
        """
        # Try to find JSON content between curly braces
        import re
        json_match = re.search(r'(\{.*\})', output, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
        
        return None
    
    def _validate_output(self, output: str, schema: Type[BaseModel]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate LLM output against a schema.
        
        Args:
            output: Output text to validate
            schema: Pydantic schema to validate against
            
        Returns:
            Tuple of (success, parsed_data)
        """
        # Try to extract JSON from the output
        parsed_json = self._parse_json_from_output(output)
        
        if not parsed_json:
            return False, None
        
        # Validate against schema
        try:
            validated = schema(**parsed_json)
            return True, validated.dict()
        except ValidationError:
            return False, None
    
    def _extract_command(self, output: str) -> str:
        """
        Extract a command from raw LLM output using various heuristics.
        
        Args:
            output: Raw LLM output text
            
        Returns:
            Extracted command
        """
        if not output or not isinstance(output, str):
            return ""
        
        # Try to extract from code blocks
        code_block_match = re.search(r"```(?:\w+)?\n(.+?)\n```", output, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # Try to extract from inline code
        inline_code_match = re.search(r"`(.+?)`", output)
        if inline_code_match:
            return inline_code_match.group(1).strip()
        
        # Try to extract from JSON
        parsed_json = self._parse_json_from_output(output)
        if parsed_json and 'command' in parsed_json:
            return parsed_json['command'].strip()
        
        # Extract the first line if it's a command
        lines = output.strip().split("\n")
        if lines and len(lines[0]) < 100 and not lines[0].startswith(('I ', 'As a')):
            return lines[0].strip()
        
        # Remove common prefixes
        prefixes_to_strip = [
            "I'd recommend", 
            "Here's the command", 
            "The command is",
            "You can use", 
            "Try using", 
            "Use this command",
            "Here is",
            "Sure,",
            "Let me",
            "To accomplish",
        ]
        
        for prefix in prefixes_to_strip:
            if output.startswith(prefix):
                cleaned = output[len(prefix):].strip()
                lines = cleaned.split("\n")
                if lines:
                    return lines[0].strip()
        
        # Return the raw output as a last resort
        return output.split("\n")[0].strip()
    
    def _sanitize_output(self, output: str) -> str:
        """
        Clean LLM output by removing explanations and non-command content.
        
        Args:
            output: Raw output text
            
        Returns:
            Sanitized output text
        """
        if not output:
            return ""
        
        # Extract command first using specialized extraction
        command = self._extract_command(output)
        
        # Remove AI self-references and explanations
        patterns = [
            r"(?i)^as an ai( language)? model[,. ]*",
            r"(?i)^as a (cybersecurity )?ai( assistant)?[,. ]*",
            r"(?i)^i am (an|a) (ai|language model)[,. ]*",
            r"(?i)^note:.*",
            r"(?i)^please note.*",
        ]
        for pat in patterns:
            command = re.sub(pat, "", command).strip()
        
        return command
    
    def _make_correction_prompt(self, original_prompt: str, output: str, schema: Type[BaseModel], error_msg: str) -> str:
        """
        Create a prompt for correcting output that failed validation.
        
        Args:
            original_prompt: Original prompt
            output: Failed output
            schema: Schema that validation failed against
            error_msg: Error message from validation
            
        Returns:
            Correction prompt
        """
        # Get the schema as JSON schema for clarity
        schema_json = schema.schema_json()
        
        return f"""Your previous response did not match the required JSON schema.
Error: {error_msg}

Original prompt: {original_prompt}

Your response was: {output}

Fix your output to match this schema exactly:
{schema_json}

Return ONLY valid JSON with no explanations or markdown formatting."""
        
    def request(self, prompt: str, role: str = "tactical", **kwargs) -> LLMResponse:
        """
        Main entry point for making LLM requests with automatic fallback.
        
        Args:
            prompt: The prompt text
            role: The role/type of request (tactical, recon, exploit, etc.)
            **kwargs: Additional parameters like context, agent_id, etc.
            
        Returns:
            LLMResponse object containing the result
        """
        self.request_count += 1
        start_time = time.time()
        
        # Parse and construct the request object
        request = LLMRequest(
            prompt=prompt,
            role=role,
            agent_id=kwargs.get('agent_id'),
            context=kwargs.get('context', {}),
            schema=kwargs.get('schema', self._get_schema_for_role(role)),
            require_validation=kwargs.get('require_validation', role in self.schema_map),
            max_tokens=kwargs.get('max_tokens'),
            temperature=kwargs.get('temperature', 0.2),
            model_preference=kwargs.get('model_preference')
        )
        
        # Try cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.in_memory_cache:
            # Parse cached response
            cached = self.in_memory_cache[cache_key]
            self.cache_hits += 1
            
            console.print(f"[cyan]🔄 Cache hit for {role} request[/cyan]")
            
            response = LLMResponse(
                content=cached["content"],
                model_used=cached["model_used"],
                tokens_used=0,  # No tokens used for cache hit
                elapsed_time=time.time() - start_time,
                validated=cached.get("validated", False),
                parsed=cached.get("parsed")
            )
            
            return response
        
        # Optimize the prompt
        optimized_prompt = self._optimize_prompt(request)
        
        # Determine the fallback chain based on task complexity
        # Simple tactical requests start with smallest model, strategic with larger models
        if role in ["tactical", "recon"] and len(prompt) < 150:
            # Start with Lily for simple tasks
            fallback_chain = [
                (LLMModelType.LOCAL_SMALL, self.lily_llm),
                (LLMModelType.LOCAL_MEDIUM, self.seneca_llm),
                (LLMModelType.CLOUD_LARGE, self.gpt_manager)
            ]
        elif role in ["exploit", "privesc"]:
            # Start with Seneca for exploit tasks
            fallback_chain = [
                (LLMModelType.LOCAL_MEDIUM, self.seneca_llm),
                (LLMModelType.LOCAL_SMALL, self.lily_llm),
                (LLMModelType.CLOUD_LARGE, self.gpt_manager)
            ]
        else:
            # Start with GPT for complex/strategic tasks
            fallback_chain = [
                (LLMModelType.CLOUD_LARGE, self.gpt_manager),
                (LLMModelType.LOCAL_MEDIUM, self.seneca_llm)
            ]
            
        # Apply model preference override if specified
        if request.model_preference:
            model_type = request.model_preference
            if model_type == LLMModelType.LOCAL_SMALL and self.lily_llm:
                fallback_chain = [(model_type, self.lily_llm)] + fallback_chain
            elif model_type == LLMModelType.LOCAL_MEDIUM and self.seneca_llm:
                fallback_chain = [(model_type, self.seneca_llm)] + fallback_chain
            elif model_type == LLMModelType.CLOUD_LARGE and self.gpt_manager:
                fallback_chain = [(model_type, self.gpt_manager)] + fallback_chain
        
        # Track attempts and results
        attempts = []
        fallbacks_used = 0
        
        # Try each model in the fallback chain
        for model_type, llm_manager in fallback_chain:
            # Skip if manager is not available
            if llm_manager is None:
                continue
                
            # Try with current model
            model_name = model_type.value
            console.print(f"[dim]🧠 Trying {model_name}...[/dim]")
            
            try:
                # Make the request with appropriate method based on manager type
                if model_type in [LLMModelType.LOCAL_SMALL, LLMModelType.LOCAL_MEDIUM]:
                    # Local models use simple querying
                    raw_output = llm_manager.query(optimized_prompt)
                else:
                    # GPT uses smart_decision with the appropriate task type
                    raw_output = llm_manager.smart_decision(
                        task_type=role,
                        task_description=optimized_prompt,
                        agent_id=request.agent_id,
                        use_gpt=True
                    )
                
                # Estimate token usage
                tokens_used = self._estimate_tokens(optimized_prompt) + self._estimate_tokens(raw_output)
                
                # Track token usage
                self.token_usage[model_name] += tokens_used
                self.token_usage["total"] += tokens_used
                
                if request.agent_id:
                    self.agent_usage[request.agent_id] = self.agent_usage.get(request.agent_id, 0) + tokens_used
                
                self.role_usage[role] = self.role_usage.get(role, 0) + tokens_used
                
                # Sanitize the output
                sanitized_output = self._sanitize_output(raw_output)
                
                # Record the attempt
                attempts.append({
                    "model": model_name,
                    "output": sanitized_output,
                    "tokens": tokens_used
                })
                
                # Validate if required
                validated = False
                parsed_data = None
                
                if request.require_validation and request.schema:
                    validated, parsed_data = self._validate_output(raw_output, request.schema)
                    
                    # If validation failed but we're not at the end of the chain,
                    # try to correct with the current model before moving to next
                    if not validated and fallbacks_used < len(fallback_chain) - 1:
                        try:
                            # Create a correction prompt
                            correction_prompt = self._make_correction_prompt(
                                optimized_prompt,
                                raw_output,
                                request.schema,
                                "Output format invalid, please fix"
                            )
                            
                            # Try again with the correction prompt
                            if model_type in [LLMModelType.LOCAL_SMALL, LLMModelType.LOCAL_MEDIUM]:
                                corrected_output = llm_manager.query(correction_prompt)
                            else:
                                corrected_output = llm_manager.smart_decision(
                                    task_type="fix",
                                    task_description=correction_prompt,
                                    agent_id=request.agent_id,
                                    use_gpt=True
                                )
                                
                            # Check if correction worked
                            validated, parsed_data = self._validate_output(corrected_output, request.schema)
                            
                            if validated:
                                console.print("[green]✓ Output corrected to match schema[/green]")
                                sanitized_output = self._sanitize_output(corrected_output)
                                
                                # Update the attempt with corrected output
                                attempts[-1]["output"] = sanitized_output
                                attempts[-1]["tokens"] += self._estimate_tokens(correction_prompt) + self._estimate_tokens(corrected_output)
                                
                                # Update token tracking
                                correction_tokens = self._estimate_tokens(correction_prompt) + self._estimate_tokens(corrected_output)
                                self.token_usage[model_name] += correction_tokens
                                self.token_usage["total"] += correction_tokens
                                
                                if request.agent_id:
                                    self.agent_usage[request.agent_id] += correction_tokens
                                
                                self.role_usage[role] += correction_tokens
                        except Exception as e:
                            console.print(f"[yellow]⚠ Correction attempt failed: {e}[/yellow]")
                            
                # If we don't need validation or the output was valid, we're done
                if not request.require_validation or validated:
                    # For structured output, extract the command if present
                    if parsed_data and "command" in parsed_data:
                        final_output = parsed_data["command"]
                    else:
                        final_output = sanitized_output
                    
                    # Save in cache
                    self.in_memory_cache[cache_key] = {
                        "content": final_output,
                        "model_used": model_name,
                        "tokens_used": tokens_used,
                        "validated": validated,
                        "parsed": parsed_data,
                        "timestamp": time.time()
                    }
                    
                    # Periodically save cache to disk
                    if self.request_count % 10 == 0:
                        threading.Thread(target=self._save_cache).start()
                    
                    # Calculate metrics
                    elapsed_time = time.time() - start_time
                    self.response_times.append(elapsed_time)
                    
                    # Create and return response
                    response = LLMResponse(
                        content=final_output,
                        model_used=model_name,
                        tokens_used=tokens_used,
                        elapsed_time=elapsed_time,
                        validated=validated,
                        parsed=parsed_data,
                        fallbacks_used=fallbacks_used
                    )
                    
                    # Log success with appropriate color based on model
                    model_colors = {
                        "lily": "cyan",
                        "seneca": "blue",
                        "gpt-nano": "magenta",
                        "gpt-main": "green"
                    }
                    color = model_colors.get(model_name, "white")
                    console.print(f"[{color}]✓ {model_name} ({'validated' if validated else 'unvalidated'}, {tokens_used} tokens, {elapsed_time:.2f}s)[/{color}]")
                    
                    return response
                
                # If validation failed, track and continue to next model
                self.validation_failures += 1
                
            except Exception as e:
                console.print(f"[yellow]⚠ {model_name} request failed: {e}[/yellow]")
            
            # If we get here, this model failed
            fallbacks_used += 1
            self.fallback_count += 1
        
        # If all models failed, return the best attempt
        console.print("[red]❌ All LLM models failed[/red]")
        
        # Find the best attempt to return
        if attempts:
            # Sort by longest output first as a simple heuristic for most complete
            best_attempt = max(attempts, key=lambda x: len(x["output"]) if x["output"] else 0)
            
            # Save in cache to avoid repeated failures
            self.in_memory_cache[cache_key] = {
                "content": best_attempt["output"],
                "model_used": best_attempt["model"],
                "tokens_used": best_attempt["tokens"],
                "validated": False,
                "parsed": None,
                "timestamp": time.time()
            }
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            self.response_times.append(elapsed_time)
            
            # Create and return response
            response = LLMResponse(
                content=best_attempt["output"] or f"Error: All {fallbacks_used} LLM models failed",
                model_used=best_attempt["model"],
                tokens_used=best_attempt["tokens"],
                elapsed_time=elapsed_time,
                validated=False,
                parsed=None,
                fallbacks_used=fallbacks_used
            )
            
            return response
        else:
            # No attempts succeeded at all
            elapsed_time = time.time() - start_time
            self.response_times.append(elapsed_time)
            
            # Return an error response
            return LLMResponse(
                content=f"Error: All LLM models failed for {role} request",
                model_used="none",
                tokens_used=0,
                elapsed_time=elapsed_time,
                validated=False,
                parsed=None,
                fallbacks_used=fallbacks_used
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance stats and metrics.
        
        Returns:
            Dict of stats and metrics
        """
        avg_time = 0
        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            
        return {
            "total_requests": self.request_count,
            "successful_requests": self.request_count - self.fallback_count,
            "fallbacks": self.fallback_count,
            "cache_hits": self.cache_hits,
            "validation_failures": self.validation_failures,
            "cache_size": len(self.in_memory_cache),
            "avg_response_time": avg_time,
            "token_usage": self.token_usage,
            "agent_usage": self.agent_usage,
            "role_usage": self.role_usage
        }
    
    def display_stats(self):
        """Display rich formatted statistics in the console."""
        stats = self.get_stats()
        
        # Create a table for token usage by model
        token_table = Table(title="Token Usage by Model")
        token_table.add_column("Model", style="cyan")
        token_table.add_column("Tokens", style="green")
        
        for model, tokens in stats["token_usage"].items():
            if model != "total":
                token_table.add_row(model, f"{tokens:,}")
        token_table.add_row("Total", f"{stats['token_usage']['total']:,}", style="bold")
            
        # Create a table for usage by role
        role_table = Table(title="Token Usage by Role")
        role_table.add_column("Role", style="cyan")
        role_table.add_column("Tokens", style="green")
        
        for role, tokens in sorted(stats["role_usage"].items(), key=lambda x: x[1], reverse=True):
            role_table.add_row(role, f"{tokens:,}")
            
        # Create a stats panel
        stats_panel = Panel(
            f"Total Requests: {stats['total_requests']}\n"
            f"Successful: {stats['successful_requests']}\n"
            f"Fallbacks: {stats['fallbacks']}\n"
            f"Cache Hits: {stats['cache_hits']}\n"
            f"Validation Failures: {stats['validation_failures']}\n"
            f"Cache Size: {stats['cache_size']} items\n"
            f"Avg Response Time: {stats['avg_response_time']:.2f}s",
            title="LLM Router Statistics"
        )
            
        # Display all tables
        console.print(stats_panel)
        console.print(token_table)
        console.print(role_table)
    
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


# CLI test mode
if __name__ == "__main__":
    console.print("[bold magenta]🚀 Testing LLM Router[/bold magenta]")
    
    router = LLMRouter()
    
    # Test simple tactical request
    response = router.request("Scan the target host for open ports", "tactical", 
                            context={"targets": ["10.10.10.10"]})
                            
    console.print(f"[green]Command:[/green] {response.content}")
    console.print(f"[blue]Model:[/blue] {response.model_used}")
    console.print(f"[blue]Tokens:[/blue] {response.tokens_used}")
    
    # Test structured output
    response = router.request("Suggest a way to exploit the SSH service on 10.10.10.10", "exploit",
                            context={"target": "10.10.10.10", "service": "ssh", "version": "OpenSSH 7.2"})
                            
    console.print(f"[green]Command:[/green] {response.content}")
    if response.parsed:
        console.print(f"[blue]Structured Output:[/blue] {response.parsed}")
    
    # Display stats
    router.display_stats()