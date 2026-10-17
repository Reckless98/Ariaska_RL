import os
for k in list(os.environ):
 if any(x in k.upper() for x in ('API_KEY','ACCESS_TOKEN','AUTH_TOKEN')):os.environ.pop(k,None)
os.environ.update(ARIASKA_DRY_RUN='1',ARIASKA_MENTOR_BACKEND='none',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',ARIASKA_LHOST='192.0.2.1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import torch
torch.set_num_threads(1)
import socket,subprocess
blocked=[]
def deny(*a,**k):
 blocked.append(str(a[0])[:80] if a else 'unknown')
 raise RuntimeError('OFFLINE_GUARD: network/process execution denied')
socket.socket.connect=deny
socket.socket.connect_ex=deny
socket.socket.sendto=deny
socket.create_connection=deny
socket.getaddrinfo=deny
subprocess.Popen=deny
os.system=deny

def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f'Offline guard blocked {len(blocked)} network/process attempts; none executed.')
