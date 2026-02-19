"""Tests for A3: Pool narrower thread safety."""
from __future__ import annotations

import os
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestPoolNarrowerThreadSafety:
    def test_import(self):
        from core.ops.pool_narrower import CommandPoolNarrower
        pn = CommandPoolNarrower()
        assert pn is not None

    def test_concurrent_record_result(self):
        from core.ops.pool_narrower import CommandPoolNarrower
        pn = CommandPoolNarrower()

        def record(thread_id):
            for i in range(50):
                pn.record_result(f"cmd_{thread_id}_{i}", success=True, reward=1.0, step=i)
                pn.record_result(f"cmd_{thread_id}_{i}", success=False, reward=-1.0, step=i)
            return thread_id

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(record, t) for t in range(4)]
            results = [f.result() for f in as_completed(futures)]
        assert len(results) == 4

    def test_concurrent_get_weighted(self):
        from core.ops.pool_narrower import CommandPoolNarrower
        pn = CommandPoolNarrower()
        candidates = [f"cmd_{i}" for i in range(20)]

        def weighted(thread_id):
            for _ in range(20):
                result = pn.get_weighted_candidates(candidates, step=10)
            return len(result)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(weighted, t) for t in range(4)]
            results = [f.result() for f in as_completed(futures)]
        assert all(r > 0 for r in results)

    def test_concurrent_reset(self):
        from core.ops.pool_narrower import CommandPoolNarrower
        pn = CommandPoolNarrower()
        # Pre-populate
        for i in range(20):
            pn.record_result(f"cmd_{i}", success=True, reward=1.0, step=i)

        def work(thread_id):
            for i in range(30):
                if i % 5 == 0:
                    pn.reset()
                pn.record_result(f"t{thread_id}_c{i}", success=True, reward=0.5, step=i)
            return True

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(work, t) for t in range(4)]
            results = [f.result() for f in as_completed(futures)]
        assert all(results)

    def test_concurrent_os_detection(self):
        from core.ops.pool_narrower import CommandPoolNarrower
        pn = CommandPoolNarrower()
        outputs = [
            "Linux version 5.4.0 GNU/Linux /bin/bash",
            "Microsoft Windows NT 10.0 cmd.exe iis/10.0",
            "OpenSSH ubuntu /etc/passwd www-data",
            "powershell.exe c:\\windows net user",
        ]

        def detect(thread_id):
            for _ in range(20):
                out = outputs[thread_id % len(outputs)]
                pn.detect_os_from_output(out)
            return True

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(detect, t) for t in range(4)]
            results = [f.result() for f in as_completed(futures)]
        assert all(results)
        # OS should be set to something
        stats = pn.get_stats()
        assert stats["detected_os"] in ("linux", "windows")

    def test_noop_lock_path(self):
        """Verify thread_safe=False uses _NoOpLock without errors."""
        from core.ops.pool_narrower import CommandPoolNarrower, NarrowerConfig
        config = NarrowerConfig(thread_safe=False)
        pn = CommandPoolNarrower(config=config)
        pn.set_target_os("linux")
        pn.record_result("nmap_scan", success=True, reward=2.0, step=1)
        weighted = pn.get_weighted_candidates(["cmd_a", "cmd_b"], step=2)
        assert len(weighted) == 2
        stats = pn.get_stats()
        assert stats["detected_os"] == "linux"
        pn.reset()
        stats2 = pn.get_stats()
        assert stats2["tracked_templates"] == 0

    def test_stats_consistency_under_contention(self):
        """Stats should reflect all recorded results even under contention."""
        from core.ops.pool_narrower import CommandPoolNarrower
        pn = CommandPoolNarrower()
        n_threads = 8
        records_per_thread = 50

        def record(tid):
            for i in range(records_per_thread):
                pn.record_result(f"shared_cmd", success=(i % 2 == 0), reward=1.0, step=i)
            return True

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(record, t) for t in range(n_threads)]
            [f.result() for f in as_completed(futures)]

        stats = pn.get_stats()
        assert stats["tracked_templates"] == 1
        # Total attempts should be n_threads * records_per_thread
        from core.ops.pool_narrower import CommandPoolNarrower as CPN
        # Access internal stats for verification
        s = pn._template_stats["shared_cmd"]
        assert s.attempts == n_threads * records_per_thread
