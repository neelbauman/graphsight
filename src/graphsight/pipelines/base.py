from abc import ABC, abstractmethod
from typing import Optional

class BasePipeline(ABC):
    """
    All execution strategies must implement this interface.
    This ensures CLI and API can treat stable/experimental logic uniformly.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        self.model_name = model

    @abstractmethod
    def run(self, image_path: str) -> str:
        """
        Takes an image path and returns Mermaid code string.
        
        Args:
            image_path (str): Path to the input image file.
            
        Returns:
            str: Generated Mermaid diagram code.
        """
        pass

