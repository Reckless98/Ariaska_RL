#!/usr/bin/env python3
"""
scripts/tools/bootstrap_tools.py — Phase 10.1B: Bootstrap Tool Installer

Checks and installs pentesting tools for a given profile (htb/ms3/ms2/dev).

Usage:
    python scripts/tools/bootstrap_tools.py --profile htb --dry-run
    python scripts/tools/bootstrap_tools.py --profile ms3 --sudo
    python scripts/tools/bootstrap_tools.py --profile dev

Outputs:
    - Console report via rich
    - JSON report to scripts/tools/bootstrap_report.json
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap pentesting tools for Ariaska")
    parser.add_argument("--profile", choices=["htb", "ms3", "ms2", "dev"], default="htb",
                       help="Tool profile to install (default: htb)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only check what would be installed")
    parser.add_argument("--sudo", action="store_true",
                       help="Allow sudo for installation")
    parser.add_argument("--output", default="scripts/tools/bootstrap_report.json",
                       help="Output report path")
    args = parser.parse_args()

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        console = Console()
    except ImportError:
        console = None

    from core.tools.tool_registry import get_tool_registry

    registry = get_tool_registry()

    if console:
        console.print(Panel(
            f"[bold]Ariaska Tool Bootstrap[/bold]\n"
            f"Profile: [cyan]{args.profile}[/cyan]  "
            f"Dry-run: [yellow]{args.dry_run}[/yellow]  "
            f"Sudo: [red]{args.sudo}[/red]",
            title="Phase 10.1B",
        ))

    report = registry.bootstrap(
        profile=args.profile,
        dry_run=args.dry_run,
        allow_sudo=args.sudo,
    )

    # Display results
    if console:
        table = Table(title=f"Bootstrap Report — {args.profile}")
        table.add_column("Tool", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")

        for name in report.skipped:
            table.add_row(name, "[green]✓ Present[/green]", "Already installed")
        for result in report.installed:
            table.add_row(result.tool_name, "[blue]✓ Installed[/blue]",
                         f"{result.method} ({result.duration_ms}ms)")
        for name in report.missing:
            table.add_row(name, "[yellow]● Missing[/yellow]", "Would install (dry-run)")
        for result in report.failures:
            table.add_row(result.tool_name, "[red]✗ Failed[/red]", result.message[:60])

        console.print(table)
        summary = report.to_dict()["summary"]
        console.print(f"\n[bold]Summary:[/bold] "
                     f"{summary['skipped']} present, "
                     f"{summary['installed']} installed, "
                     f"{summary['missing']} missing, "
                     f"{summary['failures']} failed")
    else:
        print(json.dumps(report.to_dict(), indent=2))

    # Write JSON report
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    if console:
        console.print(f"Report written to [cyan]{args.output}[/cyan]")

    return 0 if not report.failures else 1


if __name__ == "__main__":
    sys.exit(main())
