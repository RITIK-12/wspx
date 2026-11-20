"""The control plane (§3.1): API, UI, MCP endpoint, store and session lifecycle.

It runs on CPU, scales to zero, and is the store's single writer. It holds the
user's API token and the gateway admin token, never a provider key.
"""
