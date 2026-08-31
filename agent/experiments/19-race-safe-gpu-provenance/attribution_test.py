import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent.tools import _common


def main() -> None:
    root = 100
    exited_own = 201
    started_own = 202
    real_foreign = 999
    with (
        patch.object(
            _common,
            "_descendants",
            side_effect=[{root, exited_own}, {root, started_own}],
        ),
        patch.object(
            _common,
            "process_sm_utilization",
            return_value={exited_own: 98, started_own: 97, real_foreign: 96},
        ),
    ):
        sampled, own = _common.sample_process_sm_with_own(root)

    foreign = {pid: sm for pid, sm in sampled.items() if pid not in own}
    assert exited_own in own
    assert started_own in own
    assert foreign == {real_foreign: 96}
    print("PASS: before-only and after-only children excluded; foreign PID retained")


if __name__ == "__main__":
    main()
