"""Minimal no-op Retriever for fair-forge integration."""

from fair_forge.core import Retriever


class ArenaRetriever(Retriever):
    """No-op retriever — VLM Arena builds batches itself."""

    def load_dataset(self):
        return []
