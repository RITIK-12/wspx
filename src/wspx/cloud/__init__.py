"""wspx on Modal: the app, its images and the backend that runs labs as sandboxes (§5.12).

Everything is deployed into the user's own account, in a dedicated Modal
environment (``wspx``), so a Modal environment budget can cap it if wanted.
"""

APP_NAME = "wspx"
ENVIRONMENT = "wspx"
MODELS_VOLUME = "wspx-models"
STORE_VOLUME = "wspx-store"
LEDGER_VOLUME = "wspx-ledger"
CONTROL_SECRET = "wspx-control"
KEY_SECRETS = {
    "anthropic": ("wspx-key-anthropic", "ANTHROPIC_API_KEY"),
    "openai": ("wspx-key-openai", "OPENAI_API_KEY"),
    "google": ("wspx-key-google", "GEMINI_API_KEY"),
    "hf": ("wspx-key-hf", "HF_TOKEN"),
}
"""One secret per provider, so setting one key never touches the others."""
LAB_PORT = 8080
