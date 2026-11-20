"""Version control of weight edits (§5.5): objects, refs and history walks.

This layer is pure data and needs no torch, so the control plane uses it
directly. Anything that needs weights (materialising, diff magnitudes,
merges) lives in the lab and calls into it.

Objects (content-addressed JSON in the store):

``patch``   one staged edit: op name and args, precision, and its delta blobs
``tree``    what is applied on top of the base, per tensor (``TreeState``)
``commit``  parents, base, tree, the patches it introduced, result hashes,
            check report, evals, author (driver, session, calls) and message
"""

from wspx.vcs.repo import Commit, PatchRecord, Repo, short_id

__all__ = ["Commit", "PatchRecord", "Repo", "short_id"]
