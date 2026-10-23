from argparse import Namespace

import pytest

from slime.ray.rollout import _allocate_rollout_engine_addr_and_ports_normal, _group_engine_ranks_by_node_ip

NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class FakeEngine:
    def __init__(self, ip, next_port=15000):
        self.ip = ip
        self.next_port = next_port

        def _get(start_port=10000, consecutive=1):
            allocated = max(int(start_port), self.next_port)
            self.next_port = allocated + consecutive
            return self.ip, allocated

        self._get_current_node_ip_and_free_port = _RemoteMethod(_get)


@pytest.mark.unit
def test_group_engine_ranks_by_node_ip_partial_first_node():
    # 2 actor + 14 rollout on 2x8: engines 0-2 on 161, 3-6 on 165.
    grouped = _group_engine_ranks_by_node_ip(
        [
            (0, "10.161.139.161"),
            (1, "10.161.139.161"),
            (2, "10.161.139.161"),
            (3, "10.161.139.165"),
            (4, "10.161.139.165"),
            (5, "10.161.139.165"),
            (6, "10.161.139.165"),
        ]
    )
    assert grouped == [
        ("10.161.139.161", [0, 1, 2]),
        ("10.161.139.165", [3, 4, 5, 6]),
    ]


@pytest.mark.unit
def test_allocate_ports_uses_actual_node_ip_for_partial_first_node(monkeypatch):
    import slime.ray.rollout as rollout_mod

    monkeypatch.setattr(rollout_mod.ray, "get", lambda ref: ref)

    engines = (
        [(rank, FakeEngine("10.161.139.161")) for rank in range(3)]
        + [(rank, FakeEngine("10.161.139.165")) for rank in range(3, 7)]
    )
    args = Namespace(rollout_num_gpus_per_engine=2, num_gpus_per_node=8, sglang_dp_size=1)

    addr_and_ports, _ = _allocate_rollout_engine_addr_and_ports_normal(
        args=args,
        rollout_engines=engines,
        num_gpus_per_engine=2,
    )

    assert [addr_and_ports[i]["host"] for i in range(7)] == (
        ["10.161.139.161"] * 3 + ["10.161.139.165"] * 4
    )
    # Engine 3 is the first on 165 (GPU 0-1); the old rank//4 split assigned it 161.
    assert addr_and_ports[3]["host"] == "10.161.139.165"
    assert addr_and_ports[3]["port"] == 15000
