from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class BaseLLMClient(ABC):
    """Base class for LLM clients."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.client_type = kwargs.get("client")
        self.provider = kwargs.get("provider")
        self.kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
        **kwargs
    ) -> Union[Optional[str], tuple[Optional[str], Optional[Dict[str, Any]]]]:
        """Generate text from the LLM."""
        pass

    def get_context_length(self) -> int:
        """Get the context length for the model."""
        return 32768