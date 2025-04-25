#!/bin/bash
# fix_imports.sh - ARIASKA Import Path Fixer v1.0
# Fixes import path issues across the entire codebase after refactoring.

echo "🔍 ARIASKA Import Path Fixer v1.0"
echo "=================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Count variables
total_files=0
fixed_files=0

# Import patterns to fix
declare -A import_patterns=(
    ["from core\.rl_agent import RedAgent"]="from core.agents.red_agent import RedAgent"
    ["from core\.rl_agent import RLAgent"]="from core.agents.red_agent import RedAgent"
    ["from core\.models\.stats_monitor import StatsMonitor"]="from core.monitor.stats_monitor import StatsMonitor"
    ["from core\.cyber_environment import CyberEnvironment"]="from core.environment.cyber_environment import CyberEnvironment"
    ["from core\.teach import TeachModule"]="from core.teach.teach import TeachModule"
    ["from core\.value_net import ValueNet"]="from core.models.value_net import ValueNet"
    ["from core\.policy_net import PolicyNet"]="from core.models.policy_net import PolicyNet"
    ["from core\.teach\.teach\.teach import TeachModule"]="from core.teach.teach import TeachModule"
    ["from core\.memory_manager import MemoryManager"]="from core.utils.memory_manager import MemoryManager"
    ["from core\.gpt_manager import GPTManager"]="from core.gpt_manager import GPTManager"
    ["from core\.replay_buffer import ReplayBuffer"]="from core.utils.replay_buffer import ReplayBuffer"
    ["from core\.rule_engine import "]="from core.logic.rule_engine import "
    ["from core\.redundancy_detector import "]="from core.logic.redundancy_detector import "
    ["from core\.memory_router import MemoryRouter"]="from core.multiagent.memory_router import MemoryRouter"
    ["from core\.chainbuilder import "]="from core.logic.chainbuilder import "
)

# Find all Python files
echo -e "${BLUE}Searching for Python files...${NC}"
python_files=$(find . -type f -name "*.py" | grep -v "__pycache__" | grep -v "\.venv")

# Count files
for file in $python_files; do
    total_files=$((total_files+1))
done

echo -e "${GREEN}Found $total_files Python files to check.${NC}"
echo "Starting import path correction..."

# Process each file
for file in $python_files; do
    file_modified=false
    
    # Check each import pattern
    for pattern in "${!import_patterns[@]}"; do
        replacement="${import_patterns[$pattern]}"
        
        # Check if pattern exists in file
        if grep -q "$pattern" "$file"; then
            # Make backup of file
            cp "$file" "${file}.bak"
            
            # Replace the pattern
            sed -i "s|$pattern|$replacement|g" "$file"
            file_modified=true
            
            echo -e "${YELLOW}Fixed import in $file:${NC} $pattern → $replacement"
        fi
    done
    
    # If file was modified, increment counter
    if [ "$file_modified" = true ]; then
        fixed_files=$((fixed_files+1))
    fi
done

# Check for circular imports
echo -e "\n${BLUE}Checking for potential circular imports...${NC}"
circular_imports=$(grep -r "import " --include="*.py" . | grep -v "__pycache__" | sort | uniq -c | sort -nr | head -10)

echo -e "${YELLOW}Top 10 most common imports (potential circular candidates):${NC}"
echo "$circular_imports"

# Add __init__.py files where missing to ensure proper package structure
echo -e "\n${BLUE}Adding missing __init__.py files...${NC}"
dirs=$(find . -type d -not -path "*/\.*" -not -path "*/__pycache__*")

for dir in $dirs; do
    if [ ! -f "$dir/__init__.py" ]; then
        echo "# Auto-generated __init__.py" > "$dir/__init__.py"
        echo -e "${GREEN}Created $dir/__init__.py${NC}"
    fi
done

# Final summary
echo -e "\n${GREEN}Import path fixing complete!${NC}"
echo -e "Checked $total_files Python files"
echo -e "Fixed imports in $fixed_files files"
echo -e "\nIf you encounter any issues, restore from the .bak files."
echo -e "To remove backups: ${YELLOW}find . -name '*.py.bak' -delete${NC}"
