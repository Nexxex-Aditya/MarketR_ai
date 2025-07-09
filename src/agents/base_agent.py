from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import yaml
from datetime import datetime

class BaseAgent(ABC):
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the base agent.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logging()
        self.config = config or self._get_default_config()
        
    def setup_logging(self):
        """Configure logging."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'{self.__class__.__name__.lower()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {}
    
    @abstractmethod
    def process(self, data: Any) -> Dict[str, Any]:
        """
        Process the input data.
        
        Args:
            data: Input data to process
            
        Returns:
            dict: Processing results
        """
        pass 