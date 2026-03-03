#!/usr/bin/env python3
"""Parse Ariaska trace file and show step-by-step decisions."""
import json, sys

trace_file = sys.argv[1] if len(sys.argv) > 1 else "traces/events_20260227_082105.jsonl"

with open(trace_file) as f:
    for line in f:
        e = json.loads(line)
        if e.get("kind") != "step":
            continue
        s = e.get("step_num", "?")
        phase = e.get("phase_before", "?")
        total_r = e.get("step_reward_total", 0)
        
        for rec in e.get("agent_records", []):
            agent = rec.get("agent_name", "?")[:6]
            src = rec.get("decision_source", "?")
            cmd = rec.get("command", "")[:85]
            r = rec.get("reward", 0)
            disc = rec.get("discoveries", [])
            flags = rec.get("flags_set", [])
            dstr = ",".join(d.get("type", "?") if isinstance(d, dict) else str(d) for d in disc) if disc else ""
            fstr = ",".join(flags) if flags else ""
            extra = ""
            if dstr:
                extra += f" DISC=[{dstr}]"
            if fstr:
                extra += f" FLAGS=[{fstr}]"
            print(f"S{s:>2} {phase:>16} {agent:<6} src={src:<16} r={r:>6.1f}{extra}")
            print(f"    cmd: {cmd}")
