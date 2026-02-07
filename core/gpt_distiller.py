# core/gpt_distiller.py — ARIASKA GPT Distiller v2.0 APEX
# Distills agent memory and experience into compact GPT models

import os
import json
import time
import logging
import re
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from typing import Dict, List, Any, Optional

# Setup local imports
from core.gpt_manager import GPTManager
from core.multiagent.agent_manager import AgentManager

# Configure logging
logger = logging.getLogger("ariaska.distiller")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

console = Console()

class GPTDistiller:
    """
    GPT Knowledge Distillation System for ARIASKA_RL
    
    Extracts strategic insights from agent memories and refines them into concise, 
    structured knowledge that can be used to fine-tune local models or create specialized prompts.
    """
    
    def __init__(self, 
                 memory_dir: str = "core/memories", 
                 output_dir: str = "data/knowledge_sources",
                 verbosity: str = "standard"):
        """
        Initialize the GPT Distiller
        
        Args:
            memory_dir: Base directory for agent memories
            output_dir: Directory to save distilled knowledge
            verbosity: Verbosity level ('silent', 'standard', or 'verbose')
        """
        self.memory_dir = Path(memory_dir)
        self.output_dir = Path(output_dir)
        self.verbosity = verbosity
        self.gpt_manager = GPTManager()
        # All LLM functionality now handled by gpt_manager
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track stats
        self.stats = {
            "total_memories_processed": 0,
            "total_insights_generated": 0,
            "distillation_time": 0
        }

    def _log(self, message: str, level: str = "info"):
        """Log a message with the proper format and verbosity control"""
        if self.verbosity == "silent" and level not in ["error", "critical"]:
            return
            
        if level == "debug" and self.verbosity != "verbose":
            return
            
        color_map = {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "debug": "dim cyan"
        }
        color = color_map.get(level, "white")
        console.print(f"[{color}]{message}[/{color}]")
        
        # Also log to file
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, message)

    def extract_agent_memories(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Extract memories for a specific agent
        
        Args:
            agent_id: The ID of the agent
            
        Returns:
            List of memory entries
        """
        agent_memory_path = self.memory_dir / f"{agent_id.lower()}_memory"
        memories = []
        
        # Check for JSON memory files
        if agent_memory_path.exists():
            memory_files = list(agent_memory_path.glob("*.json"))
            for memory_file in memory_files:
                try:
                    with open(memory_file, "r") as f:
                        memory_data = json.load(f)
                        memories.append(memory_data)
                except Exception as e:
                    self._log(f"Error reading memory file {memory_file}: {e}", "error")
        
        # Also check logs directory for this agent
        log_path = Path("logs") / f"{agent_id}_training.log"
        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    # Extract structured data from logs if possible
                    for line in f:
                        if "[ACTION]" in line or "[REWARD]" in line:
                            # Simple parsing of log format
                            parts = line.split("|")
                            if len(parts) >= 3:
                                action = parts[1].strip() if len(parts) > 1 else "Unknown"
                                result = parts[2].strip() if len(parts) > 2 else "Unknown"
                                memories.append({
                                    "action": action,
                                    "result": result,
                                    "timestamp": parts[0].strip() if parts else "",
                                    "source": "log"
                                })
            except Exception as e:
                self._log(f"Error processing log file {log_path}: {e}", "error")
                
        self._log(f"Extracted {len(memories)} memories for {agent_id}", "debug")
        return memories

    def _generate_insight(self, memories: List[Dict[str, Any]], agent_id: str) -> Dict[str, Any]:
        """
        Generate a strategic insight from a collection of memories
        
        Args:
            memories: List of memory entries
            agent_id: The agent ID these memories belong to
            
        Returns:
            Dictionary containing the distilled insight
        """
        # Prepare context for GPT
        context_items = []
        
        # Process memories into a clean format for GPT prompt
        for mem in memories:
            if "command" in mem and "output" in mem:
                context_items.append(f"Command: {mem['command']}\nOutput: {mem['output']}")
            elif "action" in mem:
                if "result" in mem:
                    context_items.append(f"Action: {mem['action']}\nResult: {mem['result']}")
                else:
                    context_items.append(f"Action: {mem['action']}")
            elif "content" in mem:
                context_items.append(mem["content"])
                
        # Only use a subset if there are too many items (to avoid token limits)
        if len(context_items) > 12:
            # Sample a representative subset
            step = max(1, len(context_items) // 12)
            selected_items = context_items[::step][:12]
        else:
            selected_items = context_items
            
        context = "\n\n".join(selected_items)
        
        # Create a prompt for GPT to distill insights
        prompt = f"""You are ARIASKA's strategic knowledge distiller for {agent_id}.
Your task is to analyze agent memories and extract key strategic insights.
These insights will be used for agent improvement and model fine-tuning.

ANALYSIS CONTEXT:
{context}

REQUIREMENTS:
1. Identify 3-5 key tactical patterns or strategic insights.
2. For each insight, provide a short title and explanation.
3. Include any commands or techniques that proved effective.
4. Formulate a concise "strategic principle" based on each insight.

FORMAT YOUR RESPONSE AS VALID JSON with this structure:
{{
  "agent_id": "{agent_id}",
  "insights": [
    {{
      "title": "Brief descriptive title",
      "explanation": "Detailed explanation of the insight",
      "effective_techniques": ["technique1", "technique2"],
      "strategic_principle": "A concise principle derived from this insight"
    }},
    // more insights...
  ],
  "meta_strategy": "A high-level strategic recommendation for {agent_id}"
}}
"""

        # Use GPT-4o-mini for reliable structured output
        try:
            response = self.gpt_manager.gpt_request(
                prompt, 
                task_type="analysis", 
                model="gpt-5-mini"
            )
            
            # Check if the response is valid JSON
            try:
                insight = json.loads(response)
                self._log(f"Successfully generated insight using GPT-4o-mini", "success")
                return insight
            except json.JSONDecodeError:
                self._log(f"GPT response not valid JSON, attempting to extract", "warning")
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    insight = json.loads(json_match.group())
                    return insight
                else:
                    self._log(f"Could not extract valid JSON from response", "error")
                    return {"error": "Failed to parse GPT response", "raw_response": response}
        except Exception as e:
            self._log(f"Failed to generate insights: {e}", "error")
            # Provide a fallback insight structure
            return {
                "agent_id": agent_id,
                "insights": [
                    {
                        "title": "Fallback Insight",
                        "explanation": "Could not generate proper insights from memories",
                        "effective_techniques": [],
                        "strategic_principle": "When insights fail, fall back to basic techniques"
                    }
                ],
                "meta_strategy": "Review and clean agent memory data"
            }
            
    def _save_insights(self, insights: Dict[str, Any], agent_id: str):
        """Save distilled insights to file"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{agent_id.lower()}_insights_{timestamp}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, "w") as f:
                json.dump(insights, f, indent=2)
            self._log(f"Saved insights to {filepath}", "success")
            return filepath
        except Exception as e:
            self._log(f"Failed to save insights: {e}", "error")
            return None

    def distill_agent_knowledge(self, agent_id: str) -> Optional[str]:
        """
        Process all memories for an agent and distill strategic insights
        
        Args:
            agent_id: The agent ID to distill knowledge for
            
        Returns:
            Path to the saved insights file, or None if failed
        """
        self._log(f"🧠 Distilling knowledge for {agent_id}...", "info")
        
        # Extract memories
        memories = self.extract_agent_memories(agent_id)
        if not memories:
            self._log(f"No memories found for {agent_id}", "warning")
            return None
            
        self.stats["total_memories_processed"] += len(memories)
        
        # Generate insights
        insights = self._generate_insight(memories, agent_id)
        if insights:
            self.stats["total_insights_generated"] += 1
            
        # Save insights
        result = self._save_insights(insights, agent_id)
        return str(result) if result else None
        
    def distill_all_agents(self):
        """Distill knowledge for all registered agents"""
        start_time = time.time()
        
        # Use agent manager to get all agents
        try:
            agent_manager = AgentManager()
            agents = agent_manager.all_agents()
            agent_ids = [agent.agent_id for agent in agents]
            self._log(f"Found {len(agent_ids)} agents: {', '.join(agent_ids)}", "info")
        except Exception as e:
            self._log(f"Failed to get agents from AgentManager: {e}, using default list", "warning")
            # Fallback to standard agent list
            agent_ids = ["RedAgent", "BlueAgent", "ScoutAgent", "ShadowAgent", "OrionAgent"]
        
        with Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"), 
            BarColumn(), 
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(f"[cyan]Distilling knowledge for all agents...", total=len(agent_ids))
            
            for agent_id in agent_ids:
                progress.update(task, description=f"[cyan]Distilling {agent_id}...")
                self.distill_agent_knowledge(agent_id)
                progress.advance(task)
        
        # Update timing stats
        self.stats["distillation_time"] = int(time.time() - start_time)
        
        # Generate a summary report
        self._generate_summary_report()
        
        return self.stats
        
    def _generate_summary_report(self):
        """Generate and display a summary report of the distillation process"""
        console.rule("[bold green]📊 GPT Distiller Summary Report")
        
        console.print(f"[cyan]Total Memories Processed:[/cyan] {self.stats['total_memories_processed']}")
        console.print(f"[cyan]Total Insights Generated:[/cyan] {self.stats['total_insights_generated']}")
        console.print(f"[cyan]Total Time:[/cyan] {self.stats['distillation_time']:.2f} seconds")
        
        # List generated files
        if self.output_dir.exists():
            files = list(self.output_dir.glob("*_insights_*.json"))
            if files:
                console.print(f"[green]Generated {len(files)} insight files:[/green]")
                for file in files:
                    console.print(f"  - {file.name}")
            else:
                console.print("[yellow]No insight files were generated[/yellow]")
                
        # Additional info about where to find files
        console.print(f"\n[blue]Insights saved to:[/blue] {self.output_dir}")
        console.print("[blue]You can use these insights for agent improvement and model fine-tuning[/blue]")
        
def main():
    """Main entry point for the GPTDistiller when run as a script"""
    console.rule("[bold magenta]🧠 ARIASKA GPT Knowledge Distiller v2.0")
    console.print("[cyan]Starting knowledge distillation process...[/cyan]")
    
    try:
        distiller = GPTDistiller(verbosity="standard")
        distiller.distill_all_agents()
    except KeyboardInterrupt:
        console.print("\n[yellow]Distillation process interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[bold red]❌ Distillation process failed: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        return 1
        
    console.print("[green]✓ Knowledge distillation complete[/green]")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)