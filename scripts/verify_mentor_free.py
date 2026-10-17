#!/usr/bin/env python3
"""Fresh-interpreter offline proof with LLM SDK/model-library imports denied."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core.testing.offline_guard
import argparse,contextlib,importlib.abc,json,logging,os,tempfile

FORBIDDEN={'openai','anthropic','transformers','huggingface_hub','langchain'}
class NoLLMDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise ImportError('LLM dependency intentionally unavailable in offline probe')
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output.resolve()
    assert not FORBIDDEN.intersection(sys.modules)
    sys.meta_path.insert(0,NoLLMDependencies())
    with tempfile.TemporaryDirectory(prefix='ariaska-no-llm-') as temp,open(os.devnull,'w') as quiet,contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
        original=Path.cwd();os.chdir(temp);os.environ['ARIASKA_DATA_DIR']=temp+'/data';logging.disable(logging.CRITICAL)
        try:
            from core.gpt_manager import GPTManager
            from core.testing.synthetic_role_env import SyntheticRoleEnvironment
            from core.orchestration.smart_orchestrator import SmartOrchestrator,SmartOrchestratorConfig
            gm=GPTManager(enable_llm=False,offline=True)
            o=SmartOrchestrator(SyntheticRoleEnvironment(9,2),gm,
                config=SmartOrchestratorConfig(dashboard_enabled=False,dashboard_mode='off',max_steps_per_episode=2,
                    checkpoint_dir=temp+'/checkpoints'),verbosity='quiet')
            result=o.run_training(episodes=2,target_ip='192.0.2.10')
            updates={r:c.ppo_agent.updates_done for r,c in o.coaches.items()}
            assert len(updates)==5 and all(n>=2 for n in updates.values())
            assert not FORBIDDEN.intersection(sys.modules)
            assert gm.stats['total_requests']==0
        finally:os.chdir(original)
    output.write_text(json.dumps(dict(episodes=result['episodes_completed'],updates=updates,
        provider_requests=gm.stats['total_requests'],forbidden_imports_loaded=[],
        blocked_libraries=sorted(FORBIDDEN)),indent=2)+'\n')
    print(output.read_text())

if __name__=='__main__':main()
