from __future__ import annotations

from functools import wraps
from typing import Any


def apply_langchain_compatibility_patches() -> None:
    """Apply project-level compatibility patches before LangGraph imports.

    LangGraph imports ``langgraph.checkpoint.serde.jsonplus`` during agent
    startup. That module constructs ``langchain_core.load.load.Reviver()``
    without an explicit ``allowed_objects`` value, which triggers a pending
    deprecation warning.

    This project only needs LangChain message objects to be revived from
    serialized graph state, so make that default explicit before LangGraph is
    imported. This follows the dependency's migration path instead of hiding
    the warning.
    """

    from langchain_core.load.load import Reviver

    if getattr(Reviver, "_langcg_default_allowed_objects_patched", False):
        return

    original_reviver_init = Reviver.__init__

    @wraps(original_reviver_init)
    def init_with_message_allowlist(self: Reviver, *args: Any, **kwargs: Any) -> None:
        if not args and "allowed_objects" not in kwargs:
            kwargs["allowed_objects"] = "messages"
        original_reviver_init(self, *args, **kwargs)

    Reviver.__init__ = init_with_message_allowlist
    Reviver._langcg_default_allowed_objects_patched = True
