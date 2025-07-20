from core.logic.redundancy_detector import detect_redundancy_batch

def analyze_replay_buffer(buffer):
    """
    Analyze a replay buffer for redundant commands using batch redundancy detection.
    Args:
        buffer (list): List of (priority, experience) tuples from the replay buffer.
    Returns:
        list: Indices of redundant commands.
    """
    commands = [exp['command'] for _, exp in buffer if 'command' in exp]
    redundant_idxs = detect_redundancy_batch(commands)
    return redundant_idxs
