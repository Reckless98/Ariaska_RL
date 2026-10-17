"""The offline CLI exception admits only bounded Python simulation."""
from unittest.mock import Mock
import subprocess
import pytest
from core.testing import offline_guard as guard
from core.testing.guarded_cli import run_trainer_cli


@pytest.mark.parametrize('arguments', [
    ['--live'], ['--episodes','3'], ['--max-steps','4'], ['--seed','-1'],
    ['--offline','--offline'], ['--verbosity','debug'], ['--episodes'], ['-c','print(1)'], [123]])
def test_harness_rejects_unbounded_or_unknown_commands(tmp_path,monkeypatch,arguments):
    spawn=Mock();monkeypatch.setattr(guard,'_original_popen',spawn)
    with pytest.raises(ValueError):run_trainer_cli(arguments,tmp_path)
    spawn.assert_not_called()


def test_general_process_guard_is_still_installed():
    assert subprocess.Popen is guard.deny
    with pytest.raises(RuntimeError,match='OFFLINE_GUARD'):
        subprocess.run(['synthetic-unapproved-process'])
