from core.logic.redundancy_detector import detect_redundancy_batch

def analyze_replay_buffer(buffer):
    commands = [exp['command'] for _, exp in buffer if 'command' in exp]
    redundant_idxs = detect_redundancy_batch(commands)
    return redundant_idxs
