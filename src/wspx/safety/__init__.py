"""Idle compute is turned off, in independent layers (§5.14.1).

1. Session lifecycle (hot → grace → cold), in the control plane.
2. Leases: a lab runs only while it holds one; it exits when renewals stop.
3. Modal backstops: sandbox ``idle_timeout`` and ``timeout``.
4. The sweeper: every 5 minutes, terminate any wspx sandbox without a valid lease.
5. Optional Modal budgets, set by the user on the ``wspx`` environment.

``wspx kill --hard`` needs only the user's Modal token, so it works with the
control plane gone.
"""

TAG_KEY = "wspx"
TAG_VALUE = "lab"
LEASE_DICT = "wspx-leases"
