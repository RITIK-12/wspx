"""The driver runtime (§5.9): wspx's own thin agent loop over the official SDKs.

The runner holds no provider keys: every SDK client points at the LLM gateway
through ``base_url`` and carries a session token. Loop detection and any
user-set limits are enforced there and in the control plane, so a bug in the
loop cannot bypass them.
"""
