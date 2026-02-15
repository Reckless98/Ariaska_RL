#!/usr/bin/env python3
"""Run the full knowledge ingestion pipeline (Phase 9.2)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

from data.knowledge_ingestion import run_ingestion

# All repos except exploitdb (7GB gitlab repo — clone separately)
repos = [
    'hacktricks', 'payloads', 'oscp', 'h4cker', 'vulhub', 'pentest_wiki',
    'gtfobins', 'lolbas', 'wadcoms', 'atomic_red_team', 'mitre_cti',
    'internal_all_the_things', 'hacktricks_cloud', 'peass_ng', 'impacket',
    'nishang', 'ctf_writeups', 'seclists', 'metasploit',
]

manifest = run_ingestion(repos=repos)
print(f"\nDone: {manifest['total_size_mb']} MB total")
