"""The LLM gateway (§5.14.2): the only holder of provider keys.

Every paid model call passes through it. It authenticates sessions, meters
each call from the provider's usage fields, enforces limits the user has set
(none by default), detects loops, and honours the global halt flag. It is a
small proxy for each provider's native API, streaming included, with no
provider SDKs, so it stays easy to audit.
"""
