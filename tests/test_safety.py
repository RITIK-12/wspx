from wspx.safety import TAG_KEY, TAG_VALUE
from wspx.safety.sweeper import sweep


class FakeSandbox:
    def __init__(self, sid: str, tags: dict[str, str]) -> None:
        self.object_id = sid
        self.tags = tags
        self.terminated = False

    def get_tags(self) -> dict[str, str]:
        return self.tags

    def terminate(self) -> None:
        self.terminated = True


def test_sweeper_terminates_only_unleased_wspx_sandboxes() -> None:
    lab = {TAG_KEY: TAG_VALUE}
    boxes = [
        FakeSandbox("leased", lab),
        FakeSandbox("expired", lab),
        FakeSandbox("orphan", lab),
        FakeSandbox("not-ours", {"team": "x"}),
    ]
    leases = {"leased": 1000.0, "expired": 900.0}
    report = sweep(boxes, leases.get, now=1000.0, grace=30.0)
    assert sorted(report.terminated) == ["expired", "orphan"]
    assert report.kept == ["leased"]
    assert report.checked == 3
    assert not boxes[3].terminated  # never touches sandboxes wspx did not create
