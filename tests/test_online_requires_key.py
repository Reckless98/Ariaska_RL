"""Actual CLI behavior without credentials; no root .venv assumption."""
from core.testing.guarded_cli import run_trainer_cli


def test_online_without_key_fails(tmp_path):
    result = run_trainer_cli([], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'LLM' in result.stdout + result.stderr


def test_offline_without_key_succeeds(tmp_path):
    result = run_trainer_cli(['--offline'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'TRAINING SUMMARY' in result.stdout


def test_no_require_llm_flag_exists(tmp_path):
    result = run_trainer_cli(['--help'], tmp_path)
    assert result.returncode == 0
    assert '--offline' in result.stdout and '--no-require-llm' in result.stdout
