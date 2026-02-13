from abc import ABC, abstractmethod
from typing import Optional, Any

class BasePipeline(ABC):
    """
    Abstract base class for all GraphSight pipelines.
    Ensure strict separation between stable and experimental logic through this interface.
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    @abstractmethod
    def run(self, image_path: str) -> str:
        """
        Execute the pipeline to convert a flowchart image into Mermaid code.
        
        Args:
            image_path (str): Path to the input image file.
            
        Returns:
            str: Generated Mermaid diagram code.
        """
        pass

