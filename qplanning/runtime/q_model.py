# SPDX-License-Identifier: Apache-2.0
"""The scorer shares the training model and parameter names."""
from ..critic import Critic, expectation

__all__ = ["Critic", "expectation"]
