## wspx

This project edits model weights with wspx. Use the `wspx` MCP tools (or the
`wspx` CLI) rather than reading weights directly: `arch` first (the compiled map
of the model: layers, loops, exact keys), then `show` / `grep` on the weights and
`trace` / `activations` / `contrast` on the running model to find where a
behaviour lives, `edit` to stage a change, `check` and `test --vs base` to
measure it, and `vcs commit` to record it. Follow the fix lists that diagnostics
print, and never try to read held-out evaluation data.
