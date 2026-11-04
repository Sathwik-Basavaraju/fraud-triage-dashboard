"""OPTIONAL LLM client stub — DISABLED BY DEFAULT.

This project does NOT call an LLM. Alert summaries in this repo
are template-generated (see src/narrative.py).

If a future user wants LLM-written summaries, they must:
  1. Set the environment variable LLM_API_KEY.
  2. Implement `summarize_with_llm` below against their provider's
     API (endpoint, model, and prompt are intentionally NOT hard-coded
     here, so no provider is misrepresented).
  3. Change the dashboard label from "template narrative" to
     "LLM narrative" only for summaries actually returned by an LLM.

Until then, summarize_with_llm() returns None, and callers must
fall back to the template narrative.
"""

import os


def llm_enabled() -> bool:
    return bool(os.environ.get("LLM_API_KEY"))


def summarize_with_llm(template_narrative: str):
    """Return None unless an LLM is configured and implemented.

    This stub intentionally does not call any external API.
    """
    if not llm_enabled():
        return None
    raise NotImplementedError(
        "LLM summarisation is not implemented in this repo. "
        "Implement a provider call here before enabling LLM narratives."
    )
