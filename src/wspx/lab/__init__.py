"""The lab: everything that touches weights. Runs inside the GPU sandbox.

Nothing in this package holds secrets or opens network connections. It needs
the ``lab`` extra (torch, transformers, safetensors).
"""
