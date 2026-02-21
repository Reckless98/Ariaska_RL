#!/usr/bin/env python3
"""Evidence-based status report for running distillation."""
import json
import sys
import collections
import statistics
from pathlib import Path


def main() -> None:
    trace_dir = Path("/root/Ariaska_RL/traces/h200_distill")
    # Find the newest trace file
    traces = sorted(trace_dir.glob("h200_distill_*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)
    if not traces:
        print("TRACE NOT FOUND")
        sys.exit(1)
    trace = traces[0]
    print(f"Using trace: {trace.name} ({trace.stat().st_size / 1024:.1f} KB)")

    steps: list[dict] = []
    ep_starts: list[dict] = []
    ep_ends: list[dict] = []
    for line in trace.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = d.get("kind", "")
        if kind == "step":
            steps.append(d)
        elif kind == "episode_start":
            ep_starts.append(d)
        elif kind == "episode_end":
            ep_ends.append(d)

    if not steps:
        print("NO STEP DATA")
        sys.exit(1)

    total = len(steps)

    # Mentor call stats
    mentor_yes = sum(1 for s in steps if s.get("mentor_queried"))
    mentor_no = total - mentor_yes

    # Phase distribution
    phases = collections.Counter(s.get("phase", "?") for s in steps)

    # Reward stats
    rewards = [s.get("reward", 0.0) for s in steps]
    avg_r = statistics.mean(rewards)
    third = total // 3
    r_start = statistics.mean(rewards[:third]) if third > 0 else 0
    r_mid = statistics.mean(rewards[third:2 * third]) if third > 0 else 0
    r_end = statistics.mean(rewards[2 * third:]) if third > 0 else 0

    # Command families
    cmd_fams: collections.Counter[str] = collections.Counter()
    for s in steps:
        cmd = s.get("command", "")
        fam = cmd.split()[0] if cmd else "unknown"
        cmd_fams[fam] += 1

    # 5-min window mentor rates (step-based bucketing since no timestamps in step records)
    windows: dict[int, dict[str, int]] = collections.defaultdict(lambda: {"total": 0, "mentor": 0})
    for s in steps:
        w = s.get("step", 0) // 50  # bucket every 50 steps
        windows[w]["total"] += 1
        if s.get("mentor_queried"):
            windows[w]["mentor"] += 1

    # Anneal curve from episode_end records
    anneal_data = []
    for e in ep_ends:
        anneal_stage = e.get("anneal_stage", "?")
        anneal_data.append({"ep": e.get("episode", "?"), "stage": anneal_stage,
                            "mentor_calls": e.get("mentor_calls", "?")})

    print("=" * 65)
    print("  H200 DISTILLATION STATUS — EVIDENCE REPORT")
    print("=" * 65)
    print(f"  Total steps:        {total}")
    print(f"  Episodes started:   {len(ep_starts)}")
    print(f"  Episodes completed: {len(ep_ends)}")
    print(f"  Mentor YES:         {mentor_yes} ({100 * mentor_yes / total:.1f}%)")
    print(f"  Mentor NO:          {mentor_no} ({100 * mentor_no / total:.1f}%)")
    print(f"  Reward avg:         {avg_r:.3f}")
    print(f"  Reward start/mid/end: {r_start:.3f} / {r_mid:.3f} / {r_end:.3f}")
    print(f"  Reward min/max:     {min(rewards):.1f} / {max(rewards):.1f}")
    print()

    print("  Phase distribution:")
    for p, c in sorted(phases.items(), key=lambda x: -x[1]):
        print(f"    {p:<20} {c:>5} ({100 * c / total:.1f}%)")
    print()

    print("  Top 10 command families:")
    for fam, c in cmd_fams.most_common(10):
        print(f"    {fam:<25} {c:>5}")
    print()

    if anneal_data:
        print("  Anneal curve (per episode):")
        for a in anneal_data[-15:]:
            print(f"    ep={a['ep']:>3}  stage={a['stage']:<10}  mentor_calls={a['mentor_calls']}")
        if len(anneal_data) > 15:
            print(f"    ... ({len(anneal_data) - 15} earlier records omitted)")
        print()

    print("  Mentor rate per 50-step window:")
    for w in sorted(windows.keys()):
        wd = windows[w]
        rate = 100 * wd["mentor"] / wd["total"] if wd["total"] > 0 else 0
        bar = "#" * int(rate / 5)
        print(f"    window {w:>3}: {wd['mentor']:>4}/{wd['total']:>4} = {rate:>5.1f}% |{bar}")
    print()

    print("  Last 5 steps:")
    for s in steps[-5:]:
        print(f"    ep={s.get('episode', '?'):>3} step={s.get('step', '?'):>4} "
              f"phase={s.get('phase', '?'):<14} reward={s.get('reward', 0):>6.1f} "
              f"mentor={s.get('mentor_queried', '?')}")
    print()

    if ep_ends:
        print("  Last 5 completed episodes:")
        for e in ep_ends[-5:]:
            print(f"    ep={e.get('episode', '?'):>3}  reward={e.get('total_reward', 0):>8.1f}  "
                  f"steps={e.get('steps', '?'):>4}  max_phase={e.get('max_phase', '?'):<15}  "
                  f"mentor={e.get('mentor_calls', '?')}")
    print()

    # Checkpoints
    ckpt_dir = Path("/root/Ariaska_RL/models/distilled")
    ckpts = sorted(ckpt_dir.glob("h200_*.pt"))
    if ckpts:
        print(f"  Checkpoints ({len(ckpts)} total):")
        for c in ckpts:
            print(f"    {c.name}  ({c.stat().st_size / 1024 / 1024:.1f} MB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
