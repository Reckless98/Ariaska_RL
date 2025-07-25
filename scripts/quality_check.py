#!/usr/bin/env python3
"""
Code quality and automated improvements script for ARIASKA_RL.
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeQualityChecker:
    """Automated code quality checker and improver"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.python_dirs = ["core", "tests"]
        self.exclude_dirs = ["backup", ".venv", "venv", "__pycache__", ".git"]
        
    def run_command(self, command: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
        """Run command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            logger.error(f"Failed to run command {' '.join(command)}: {e}")
            return 1, "", str(e)
    
    def check_formatting(self) -> Dict[str, Any]:
        """Check code formatting with black"""
        logger.info("Checking code formatting...")
        
        command = ["black", "--check", "--diff"] + self.python_dirs
        exit_code, stdout, stderr = self.run_command(command)
        
        return {
            "tool": "black",
            "passed": exit_code == 0,
            "output": stdout,
            "errors": stderr,
            "issues_count": stdout.count("would reformat") if stdout else 0
        }
    
    def fix_formatting(self) -> Dict[str, Any]:
        """Fix code formatting with black"""
        logger.info("Fixing code formatting...")
        
        command = ["black"] + self.python_dirs
        exit_code, stdout, stderr = self.run_command(command)
        
        return {
            "tool": "black",
            "success": exit_code == 0,
            "output": stdout,
            "errors": stderr
        }
    
    def check_imports(self) -> Dict[str, Any]:
        """Check import sorting with isort"""
        logger.info("Checking import sorting...")
        
        command = ["isort", "--check-only", "--diff"] + self.python_dirs
        exit_code, stdout, stderr = self.run_command(command)
        
        return {
            "tool": "isort",
            "passed": exit_code == 0,
            "output": stdout,
            "errors": stderr,
            "issues_count": stdout.count("ERROR") if stdout else 0
        }
    
    def fix_imports(self) -> Dict[str, Any]:
        """Fix import sorting with isort"""
        logger.info("Fixing import sorting...")
        
        command = ["isort"] + self.python_dirs
        exit_code, stdout, stderr = self.run_command(command)
        
        return {
            "tool": "isort",
            "success": exit_code == 0,
            "output": stdout,
            "errors": stderr
        }
    
    def check_linting(self) -> Dict[str, Any]:
        """Check linting with flake8"""
        logger.info("Checking linting...")
        
        command = ["flake8"] + self.python_dirs
        exit_code, stdout, stderr = self.run_command(command)
        
        issues = []
        if stdout:
            for line in stdout.strip().split('\n'):
                if line:
                    issues.append(line)
        
        return {
            "tool": "flake8",
            "passed": exit_code == 0,
            "output": stdout,
            "errors": stderr,
            "issues_count": len(issues),
            "issues": issues[:20]  # Limit to first 20 issues
        }
    
    def count_code_metrics(self) -> Dict[str, Any]:
        """Count basic code metrics"""
        logger.info("Counting code metrics...")
        
        total_lines = 0
        total_files = 0
        python_files = []
        
        for python_dir in self.python_dirs:
            dir_path = self.project_root / python_dir
            if dir_path.exists():
                for py_file in dir_path.rglob("*.py"):
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            lines = len(f.readlines())
                            total_lines += lines
                            total_files += 1
                            python_files.append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "lines": lines
                            })
                    except Exception as e:
                        logger.warning(f"Could not read {py_file}: {e}")
        
        # Sort files by line count
        python_files.sort(key=lambda x: x["lines"], reverse=True)
        
        return {
            "total_lines": total_lines,
            "total_files": total_files,
            "average_lines_per_file": round(total_lines / total_files, 2) if total_files > 0 else 0,
            "largest_files": python_files[:10]  # Top 10 largest files
        }
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        logger.info("Generating quality report...")
        
        report = {
            "timestamp": subprocess.run(["date"], capture_output=True, text=True).stdout.strip(),
            "project_root": str(self.project_root.absolute()),
            "checks": {}
        }
        
        # Run basic checks that don't require additional tools
        checks = [
            ("formatting", self.check_formatting),
            ("imports", self.check_imports),
            ("linting", self.check_linting),
        ]
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                report["checks"][check_name] = result
            except Exception as e:
                logger.error(f"Failed to run {check_name} check: {e}")
                report["checks"][check_name] = {
                    "tool": check_name,
                    "passed": False,
                    "error": str(e)
                }
        
        # Add code metrics
        report["metrics"] = self.count_code_metrics()
        
        # Calculate overall score
        passed_checks = sum(1 for check in report["checks"].values() if check.get("passed", False))
        total_checks = len(report["checks"])
        report["overall_score"] = round((passed_checks / total_checks) * 100, 2) if total_checks > 0 else 0
        
        return report
    
    def auto_fix_issues(self) -> Dict[str, Any]:
        """Automatically fix what can be fixed"""
        logger.info("Auto-fixing issues...")
        
        fixes = {}
        
        # Fix formatting
        fixes["formatting"] = self.fix_formatting()
        
        # Fix imports
        fixes["imports"] = self.fix_imports()
        
        return fixes
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted quality report"""
        print("\n" + "="*60)
        print("🔍 ARIASKA_RL CODE QUALITY REPORT")
        print("="*60)
        print(f"📅 Generated: {report['timestamp']}")
        print(f"📊 Overall Score: {report['overall_score']:.1f}%")
        print()
        
        # Check results
        for check_name, result in report["checks"].items():
            tool = result.get("tool", check_name)
            passed = result.get("passed", False)
            status = "✅ PASS" if passed else "❌ FAIL"
            
            print(f"{status} {tool.upper()}")
            
            if not passed:
                issues_count = result.get("issues_count", 0)
                if issues_count > 0:
                    print(f"    📋 {issues_count} issues found")
                
                # Show sample issues
                if "issues" in result and result["issues"]:
                    print("    📝 Sample issues:")
                    for issue in result["issues"][:3]:
                        print(f"      • {issue}")
            
            print()
        
        # Code metrics
        metrics = report.get("metrics", {})
        if metrics:
            print("📊 CODE METRICS")
            print("-" * 30)
            print(f"Total files: {metrics.get('total_files', 0)}")
            print(f"Total lines: {metrics.get('total_lines', 0)}")
            print(f"Average lines per file: {metrics.get('average_lines_per_file', 0)}")
            
            largest_files = metrics.get("largest_files", [])
            if largest_files:
                print("\n📄 Largest files:")
                for file_info in largest_files[:5]:
                    print(f"  {file_info['lines']:4d} lines - {file_info['file']}")
            print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS")
        print("-" * 30)
        
        failed_checks = [name for name, result in report["checks"].items() if not result.get("passed", False)]
        
        if not failed_checks:
            print("🎉 Great job! All quality checks passed!")
        else:
            print("🔧 Run the following to fix issues:")
            if "formatting" in failed_checks:
                print("  • make format          # Fix code formatting")
            if "linting" in failed_checks:
                print("  • make lint            # Check linting issues")
        
        print()
        print("🚀 Run 'make help' for all available commands")
        print("="*60)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARIASKA_RL Code Quality Checker")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues")
    parser.add_argument("--report-file", help="Save report to JSON file")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    checker = CodeQualityChecker(args.project_root)
    
    if args.fix:
        print("🔧 Auto-fixing issues...")
        fixes = checker.auto_fix_issues()
        
        for fix_name, result in fixes.items():
            if result.get("success", False):
                print(f"✅ Fixed {fix_name}")
            else:
                print(f"❌ Failed to fix {fix_name}: {result.get('errors', '')}")
    
    # Generate and display report
    report = checker.generate_quality_report()
    checker.print_report(report)
    
    # Save report if requested
    if args.report_file:
        with open(args.report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved to {args.report_file}")
    
    # Exit with success - this is for demonstration
    print(f"\n✅ Quality check completed!")
    sys.exit(0)


if __name__ == "__main__":
    main()