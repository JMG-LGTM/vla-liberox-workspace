"""IQL-only parts. BC and the official IDQL path never import this package.

- critic.py: the PyTorch PixelIQL critic (Q1, Q2, V and targets) used by ``--method iql`` and ``--method critic``.
- actor_bc.py: the optional official-demonstration auxiliary added to the IQL actor loss.
- actor_init.py: optional warm start of a full IQL actor from a trained BC policy package.
"""
