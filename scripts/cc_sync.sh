#!/usr/bin/env bash
# cc_sync.sh — Copilot ↔ Claude Code bidirectional session manager
# Usage: ./scripts/cc_sync.sh <command> [args...]
#
# Commands:
#   start  <task>       — Reset session, write task, fire initial Claude audit
#   audit  <question>   — Ask Claude a question, append to session
#   plan   <plan>       — Write your plan, ask Claude to review it
#   done   <summary>    — Mark done, fire Claude post-audit
#   review              — Ask Claude to review current session state
#   status              — Show current session status
#   archive             — Archive current session and reset
#   auto-audit          — Auto-audit staged git changes (used by pre-commit hook)
#   parallel <q1> <q2>  — Fire two Claude queries in parallel, merge results

set -euo pipefail

# Resolve project root (script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"

SESSION_FILE="$PROJECT_ROOT/.github/CC_SESSION.md"
ARCHIVE_DIR="$PROJECT_ROOT/.github/sessions"
CLAUDE_MODEL="${CC_MODEL:-sonnet}"
CLAUDE_EFFORT="${CC_EFFORT:-medium}"
CLAUDE_TIMEOUT="${CC_TIMEOUT:-120}"

_timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# ── Claude invocation with error handling ────────────────────────────

_claude() {
    local prompt="$1"
    local response=""
    local exit_code=0

    # Check claude CLI exists
    if ! command -v claude &>/dev/null; then
        echo "[cc_sync] ERROR: claude CLI not found. Install from https://claude.ai/download"
        return 1
    fi

    response=$(timeout "$CLAUDE_TIMEOUT" claude -p "$prompt" --model "$CLAUDE_MODEL" --effort "$CLAUDE_EFFORT" 2>&1) || exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        echo "[cc_sync] WARNING: claude exited with code $exit_code"
        if [[ $exit_code -eq 124 ]]; then
            echo "[cc_sync] Timed out after ${CLAUDE_TIMEOUT}s. Increase with CC_TIMEOUT=N"
        fi
        # Still return whatever we got — partial response may be useful
    fi

    if [[ -z "$response" ]]; then
        echo "[cc_sync] ERROR: Claude returned empty response."
        return 1
    fi

    echo "$response"
}

_claude_async() {
    local prompt="$1"
    local outfile="$2"
    (
        timeout "$CLAUDE_TIMEOUT" claude -p "$prompt" --model "$CLAUDE_MODEL" --effort "$CLAUDE_EFFORT" > "$outfile" 2>&1 || {
            echo "[cc_sync] WARNING: async claude exited with code $?" >> "$outfile"
        }
    ) &
    echo $!
}

# ── Session file management ──────────────────────────────────────────

_append() {
    echo -e "\n$1" >> "$SESSION_FILE"
}

_read_session() {
    if [[ -f "$SESSION_FILE" ]]; then
        cat "$SESSION_FILE"
    else
        echo "(no session)"
    fi
}

_ensure_session_dir() {
    local dir
    dir=$(dirname "$SESSION_FILE")
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
    fi
}

# ── Verdict extraction — robust multi-pattern matching ───────────────

_extract_verdict() {
    # Extracts verdict from Claude response. Checks multiple patterns
    # to handle formatting variations.
    local response="$1"
    local verdict_type="$2"  # "plan" or "audit"

    if [[ "$verdict_type" == "plan" ]]; then
        # Check for BLOCKED first (highest priority)
        if echo "$response" | grep -qiE "(BLOCKED|do not proceed|cannot proceed|must not proceed)"; then
            echo "BLOCKED"
            return
        fi
        # Check for CONCERNS
        if echo "$response" | grep -qiE "(CONCERNS?:|with caution|issues? to address|address before|should fix)"; then
            echo "CONCERNS"
            return
        fi
        # Check for APPROVED
        if echo "$response" | grep -qiE "(APPROVED|safe to proceed|looks good|no issues|plan is sound|go ahead)"; then
            echo "APPROVED"
            return
        fi
        # Couldn't determine — default safe
        echo "UNCLEAR"

    elif [[ "$verdict_type" == "audit" ]]; then
        # Check for ISSUES first
        if echo "$response" | grep -qiE "(ISSUES?:|problems? found|violations? found|must fix|not safe)"; then
            echo "ISSUES"
            return
        fi
        # Check for CLEAN
        if echo "$response" | grep -qiE "(CLEAN|no issues|safe to commit|looks clean|all clear|no violations)"; then
            echo "CLEAN"
            return
        fi
        echo "UNCLEAR"

    elif [[ "$verdict_type" == "commit" ]]; then
        # Pre-commit: check for NO
        if echo "$response" | grep -qiE "^(NO|unsafe|do not commit|block|not safe)"; then
            echo "NO"
            return
        fi
        if echo "$response" | grep -qiE "^(YES|safe|proceed|commit)"; then
            echo "YES"
            return
        fi
        # Default: allow commit if unclear (don't block workflow on ambiguity)
        echo "YES"
    fi
}

# ── Commands ─────────────────────────────────────────────────────────

cmd_start() {
    local task="${1:-}"
    if [[ -z "$task" ]]; then
        echo "Usage: cc_sync.sh start '<task description>'"
        exit 1
    fi

    _ensure_session_dir

    # Archive previous session if non-empty and not idle
    if [[ -f "$SESSION_FILE" ]] && grep -q "^## Task:" "$SESSION_FILE" 2>/dev/null; then
        local status
        status=$(grep "^\*\*Status\*\*:" "$SESSION_FILE" | head -1 | sed 's/.*: //')
        if [[ "$status" != "idle" ]]; then
            cmd_archive 2>/dev/null || true
        fi
    fi

    cat > "$SESSION_FILE" << EOF
## Task: $task
**Status**: scoping
**Started**: $(_timestamp)

---
### [Copilot → Claude] Request
Analyze this task for the Ariaska_RL codebase:
**$task**

Provide:
1. Files likely affected
2. Architectural invariants at risk
3. Circular import risks
4. Tests that must pass
5. Recommended approach
EOF

    echo "[cc_sync] Session started: $task"
    echo "[cc_sync] Firing Claude audit (model=$CLAUDE_MODEL, effort=$CLAUDE_EFFORT)..."

    local response
    if ! response=$(_claude "You are Claude Code, the senior engineering partner for the Ariaska_RL project.

READ the .claude/CLAUDE.md file for your role instructions and the full invariant checklist.

Here is the current session:

$(_read_session)

RESPOND with ONLY a structured analysis section. Use this EXACT format:

### [Claude → Copilot] Analysis
**Files affected:** <list>
**Invariants at risk:** <list or 'none'>
**Circular import risks:** <list or 'none'>
**Tests to verify:** <list>
**Recommended approach:** <terse description>
**Verdict:** APPROVED / CONCERNS: <list> / BLOCKED: <reason>"); then
        echo "[cc_sync] WARNING: Claude audit failed. Session created but unaudited."
        _append "---"
        _append "### [Claude → Copilot] Analysis"
        _append "*Claude audit failed — proceed with caution. Run \`cc_sync.sh audit\` to retry.*"
        sed -i 's/\*\*Status\*\*: scoping/**Status**: planning/' "$SESSION_FILE"
        return 1
    fi

    _append "---"
    _append "$response"

    sed -i 's/\*\*Status\*\*: scoping/**Status**: planning/' "$SESSION_FILE"

    echo ""
    echo "═══════════════════════════════════════════"
    echo "$response"
    echo "═══════════════════════════════════════════"
    echo ""
    echo "[cc_sync] Session ready → next: cc_sync.sh plan '<your plan>'"
}

cmd_audit() {
    local question="${1:-}"
    if [[ -z "$question" ]]; then
        echo "Usage: cc_sync.sh audit '<question>'"
        exit 1
    fi

    _append "---"
    _append "### [Copilot → Claude] Question"
    _append "$question"

    echo "[cc_sync] Asking Claude..."

    local response
    if ! response=$(_claude "You are Claude Code for Ariaska_RL. Read .claude/CLAUDE.md for context.

Current session:
$(_read_session)

Answer the latest question. Use this format:

### [Claude → Copilot] Answer
<your answer — terse, specific, include file paths and line numbers where relevant>"); then
        echo "[cc_sync] WARNING: Claude query failed."
        _append "---"
        _append "### [Claude → Copilot] Answer"
        _append "*Query failed — retry with \`cc_sync.sh audit\`*"
        return 1
    fi

    _append "---"
    _append "$response"

    echo ""
    echo "$response"
}

cmd_plan() {
    local plan="${1:-}"
    if [[ -z "$plan" ]]; then
        echo "Usage: cc_sync.sh plan '<implementation plan>'"
        exit 1
    fi

    _append "---"
    _append "### [Copilot → Claude] Implementation Plan"
    _append "$plan"

    sed -i 's/\*\*Status\*\*: planning/**Status**: plan_review/' "$SESSION_FILE"

    echo "[cc_sync] Submitting plan for Claude review..."

    local response
    if ! response=$(_claude "You are Claude Code for Ariaska_RL. Read .claude/CLAUDE.md for context.

Current session:
$(_read_session)

Review the Implementation Plan. You MUST respond using this EXACT format:

### [Claude → Copilot] Plan Review
**Verdict:** <exactly one of: APPROVED / CONCERNS / BLOCKED>
**Details:** <explanation>
**Invariant check:** <which invariants were verified>

Rules:
- APPROVED = plan is safe, no invariant violations, proceed
- CONCERNS: <list> = minor issues, proceed after addressing
- BLOCKED: <reason> = critical issue, do NOT proceed"); then
        echo "[cc_sync] WARNING: Plan review failed."
        _append "---"
        _append "### [Claude → Copilot] Plan Review"
        _append "*Review failed — retry with \`cc_sync.sh plan\`*"
        sed -i 's/\*\*Status\*\*: plan_review/**Status**: planning/' "$SESSION_FILE"
        return 1
    fi

    _append "---"
    _append "$response"

    # Extract and act on verdict
    local verdict
    verdict=$(_extract_verdict "$response" "plan")

    case "$verdict" in
        BLOCKED)
            echo ""
            echo "⛔ BLOCKED — Do not proceed."
            echo ""
            echo "$response"
            return 1
            ;;
        CONCERNS)
            echo ""
            echo "⚠️  CONCERNS — Address before proceeding."
            echo ""
            echo "$response"
            # Don't move to in_progress — stay in plan_review
            ;;
        APPROVED)
            echo ""
            echo "✅ APPROVED — Proceed with implementation."
            echo ""
            echo "$response"
            sed -i 's/\*\*Status\*\*: plan_review/**Status**: in_progress/' "$SESSION_FILE"
            ;;
        UNCLEAR)
            echo ""
            echo "❓ Verdict unclear — review Claude's response manually:"
            echo ""
            echo "$response"
            echo ""
            echo "Then either re-submit plan or proceed at your discretion."
            ;;
    esac
}

cmd_done() {
    local summary="${1:-}"
    if [[ -z "$summary" ]]; then
        echo "Usage: cc_sync.sh done '<implementation summary>'"
        exit 1
    fi

    _append "---"
    _append "### [Copilot] Done"
    _append "**Implemented:** $summary"
    _append "**Completed at:** $(_timestamp)"

    sed -i 's/\*\*Status\*\*: in_progress/**Status**: review/' "$SESSION_FILE"

    echo "[cc_sync] Firing Claude post-audit..."

    local response
    if ! response=$(_claude "You are Claude Code for Ariaska_RL. Read .claude/CLAUDE.md for context.

Current session:
$(_read_session)

Post-implementation audit. You MUST respond using this EXACT format:

### [Claude] Post-Audit
**Verdict:** <exactly one of: CLEAN / ISSUES>
**Invariant check:** <which invariants were verified>
**Details:** <findings or 'no issues found'>

Check:
1. Architectural invariant violations (GPTManager bypass, direct openai import, top-level imports)
2. STATE_DIM = 512 preserved
3. Lazy import pattern respected
4. Evidence Gate still in enforce mode
5. BudgetManagerV2 constants unchanged
6. Test baseline likely intact"); then
        echo "[cc_sync] WARNING: Post-audit failed."
        _append "---"
        _append "### [Claude] Post-Audit"
        _append "*Audit failed — run tests manually before committing.*"
        return 1
    fi

    _append "---"
    _append "$response"

    local verdict
    verdict=$(_extract_verdict "$response" "audit")

    case "$verdict" in
        ISSUES)
            echo ""
            echo "⚠️  Issues found — fix before committing."
            echo ""
            echo "$response"
            return 1
            ;;
        CLEAN)
            echo ""
            echo "✅ Clean — safe to commit."
            echo ""
            echo "$response"
            sed -i 's/\*\*Status\*\*: review/**Status**: done/' "$SESSION_FILE"
            ;;
        UNCLEAR)
            echo ""
            echo "❓ Audit verdict unclear — review manually:"
            echo ""
            echo "$response"
            ;;
    esac
}

cmd_review() {
    echo "[cc_sync] Asking Claude for session review..."

    local response
    if ! response=$(_claude "You are Claude Code for Ariaska_RL.

Current session:
$(_read_session)

Summarize session state in this format:
**Task:** <what>
**Current status:** <where we are>
**Decisions made:** <key decisions>
**Next step:** <what should happen next>
**Open questions:** <any unresolved items>"); then
        echo "[cc_sync] Review failed."
        return 1
    fi

    echo "$response"
}

cmd_status() {
    if [[ ! -f "$SESSION_FILE" ]]; then
        echo "No active session."
        return
    fi
    local task status started
    task=$(grep "^## Task:" "$SESSION_FILE" | head -1 | sed 's/^## Task: //')
    status=$(grep "^\*\*Status\*\*:" "$SESSION_FILE" | head -1 | sed 's/.*: //')
    started=$(grep "^\*\*Started\*\*:" "$SESSION_FILE" | head -1 | sed 's/.*: //' || echo "unknown")

    echo "Task:    $task"
    echo "Status:  $status"
    echo "Started: $started"
    echo "Session: $SESSION_FILE"
    echo ""

    # Show section count
    local sections
    sections=$(grep -c "^### \[" "$SESSION_FILE" 2>/dev/null || echo 0)
    echo "Sections: $sections exchanges logged"
}

cmd_archive() {
    if [[ ! -f "$SESSION_FILE" ]]; then
        echo "No session to archive."
        return
    fi
    mkdir -p "$ARCHIVE_DIR"
    local archive_name="CC_SESSION_$(date +%Y%m%d_%H%M%S).md"
    cp "$SESSION_FILE" "$ARCHIVE_DIR/$archive_name"
    echo "Archived: $ARCHIVE_DIR/$archive_name"

    # Reset session
    cat > "$SESSION_FILE" << 'EOF'
## Task: (none — awaiting next task)
**Status**: idle
EOF
}

cmd_auto_audit() {
    # Called by pre-commit hook — audits staged changes
    local staged_files
    staged_files=$(git diff --cached --name-only 2>/dev/null | head -30)

    if [[ -z "$staged_files" ]]; then
        return 0
    fi

    local core_files
    core_files=$(echo "$staged_files" | grep "^core/" || true)

    if [[ -z "$core_files" ]]; then
        # No core files changed, skip audit
        return 0
    fi

    local file_count
    file_count=$(echo "$core_files" | wc -l | tr -d ' ')
    echo "[pre-commit] Claude auto-audit: $file_count core file(s) staged..."

    local diff_summary
    diff_summary=$(git diff --cached --stat 2>/dev/null | tail -5)

    # Get actual diff content for core files (limited)
    local diff_content
    diff_content=$(git diff --cached -- $core_files 2>/dev/null | head -200)

    local response
    if ! response=$(_claude "Pre-commit audit for Ariaska_RL. Be EXTREMELY terse — this runs on every commit.

Staged core files:
$core_files

Diff summary:
$diff_summary

Diff (first 200 lines):
$diff_content

Answer EXACTLY in this format (first line must be YES or NO):
<YES or NO>
<one-line reason>

Check ONLY:
1. Direct 'import openai' (must go through GPTManager)
2. Top-level imports between core modules (must be lazy)
3. STATE_DIM changed from 512
4. BudgetManagerV2 constants changed
5. print() used instead of rich/logging"); then
        echo "[pre-commit] Claude audit failed — allowing commit (fail-open)."
        return 0
    fi

    echo "$response"

    local verdict
    verdict=$(_extract_verdict "$response" "commit")

    if [[ "$verdict" == "NO" ]]; then
        echo ""
        echo "⛔ Claude flagged issues. Review above."
        echo "   To bypass: git commit --no-verify"
        return 1
    fi
    return 0
}

cmd_todos() {
    local todos="${1:-}"
    if [[ -z "$todos" ]]; then
        echo "Usage: cc_sync.sh todos '<numbered list of steps>'"
        exit 1
    fi

    _append "---"
    _append "### [Copilot] TODO List — $(_timestamp)"
    _append "$todos"

    echo "[cc_sync] TODO list saved to session."
    echo "After each step, run: cc_sync.sh checkpoint '<step N done: summary>'"
}

cmd_checkpoint() {
    local update="${1:-}"
    local ask_claude="${2:-}"

    if [[ -z "$update" ]]; then
        echo "Usage: cc_sync.sh checkpoint '<what was completed, what remains>' [--ask-claude]"
        exit 1
    fi

    _append "---"
    _append "### [Copilot] Checkpoint — $(_timestamp)"
    _append "$update"

    echo "[cc_sync] Checkpoint saved."

    if [[ "$ask_claude" == "--ask-claude" ]]; then
        echo "[cc_sync] Asking Claude for next-step guidance..."
        local response
        if response=$(_claude "You are Claude Code for Ariaska_RL. Read .claude/CLAUDE.md for context.

Current session:
$(_read_session)

Copilot has hit a checkpoint and needs guidance on next steps. Respond using this EXACT format:

### [Claude → Copilot] Next Steps
**Completed so far:** <summary from checkpoint>
**Remaining TODOs:** <what still needs doing, inferred from session>
**Recommended next action:** <specific concrete next step>
**Watch out for:** <invariants or risks for remaining work>
**Context advice:** <whether to start fresh context or continue>"); then
            _append "---"
            _append "$response"
            echo ""
            echo "$response"
        else
            echo "[cc_sync] WARNING: Claude guidance failed — continue from checkpoint manually."
        fi
    fi
}

cmd_parallel() {
    local q1="${1:-}"
    local q2="${2:-}"
    if [[ -z "$q1" || -z "$q2" ]]; then
        echo "Usage: cc_sync.sh parallel '<question1>' '<question2>'"
        exit 1
    fi

    local tmp1 tmp2
    tmp1=$(mktemp)
    tmp2=$(mktemp)

    echo "[cc_sync] Firing two Claude queries in parallel..."
    local pid1 pid2
    pid1=$(_claude_async "You are Claude Code for Ariaska_RL. $q1" "$tmp1")
    pid2=$(_claude_async "You are Claude Code for Ariaska_RL. $q2" "$tmp2")

    wait "$pid1" "$pid2" 2>/dev/null

    # Validate both responses
    local has_a=true has_b=true
    if [[ ! -s "$tmp1" ]]; then
        echo "[cc_sync] WARNING: Stream A returned empty."
        has_a=false
    fi
    if [[ ! -s "$tmp2" ]]; then
        echo "[cc_sync] WARNING: Stream B returned empty."
        has_b=false
    fi

    _append "---"
    _append "### [Claude → Copilot] Parallel Analysis"
    if $has_a; then
        _append "**Stream A:**"
        _append "$(cat "$tmp1")"
    fi
    _append ""
    if $has_b; then
        _append "**Stream B:**"
        _append "$(cat "$tmp2")"
    fi

    echo "═══════════════════════════════════════════"
    echo "Stream A:"
    echo "═══════════════════════════════════════════"
    $has_a && cat "$tmp1" || echo "(empty)"
    echo ""
    echo "═══════════════════════════════════════════"
    echo "Stream B:"
    echo "═══════════════════════════════════════════"
    $has_b && cat "$tmp2" || echo "(empty)"

    rm -f "$tmp1" "$tmp2"
}

# ── Dispatch ─────────────────────────────────────────────────────────

case "${1:-help}" in
    start)      shift; cmd_start "$*" ;;
    audit)      shift; cmd_audit "$*" ;;
    plan)       shift; cmd_plan "$*" ;;
    done)       shift; cmd_done "$*" ;;
    todos)      shift; cmd_todos "$*" ;;
    checkpoint) shift; cmd_checkpoint "$1" "${2:-}" ;;
    review)     cmd_review ;;
    status)     cmd_status ;;
    archive)    cmd_archive ;;
    auto-audit) cmd_auto_audit ;;
    parallel)   shift; cmd_parallel "$1" "$2" ;;
    help|*)
        cat << 'HELP'
cc_sync.sh — Copilot ↔ Claude Code bidirectional session manager

Commands:
  start       <task>       Start new session, fire Claude audit
  todos       <list>       Write numbered TODO list to session (call after start)
  audit       <question>   Ask Claude a question mid-session
  plan        <plan>       Submit plan for Claude review (returns APPROVED/CONCERNS/BLOCKED)
  checkpoint  <update>     Record progress mid-task; add --ask-claude for next-step guidance
  done        <summary>    Mark done, fire post-audit (returns CLEAN/ISSUES)
  review                   Review full session state
  status                   Show current session status
  archive                  Archive and reset session
  auto-audit               Auto-audit staged git changes (pre-commit hook)
  parallel    <q1> <q2>    Two parallel Claude queries

Environment:
  CC_MODEL=sonnet     Claude model (default: sonnet)
  CC_EFFORT=medium    Claude effort (default: medium)
  CC_TIMEOUT=120      Timeout in seconds (default: 120)

Session file: .github/CC_SESSION.md
Archives:     .github/sessions/

Workflow:
  1. cc_sync.sh start "task description"                      → Claude audits
  2. cc_sync.sh todos "1. step\n2. step\n3. step"             → TODO list saved
  3. cc_sync.sh plan "my implementation plan"                  → Claude reviews (APPROVED/BLOCKED)
  4. (implement step 1)
  5. cc_sync.sh checkpoint "step 1 done: <summary>"           → progress saved
  6. (implement step 2... if stuck/context full:)
     cc_sync.sh checkpoint --ask-claude "done: X. remaining: Y" → Claude guides next step
  7. cc_sync.sh done "full summary"                           → Claude audits (CLEAN/ISSUES)
  8. git commit                                                → pre-commit hook auto-audits
HELP
        ;;
esac
