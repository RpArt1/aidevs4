from abc import ABC
from abc import abstractmethod
from common import LLMService, AssignmentService, get_logger
from common import setup_logging


class Assignment(ABC):
    def __init__(self, name: str, description: str ) : 
        self.name = name
        self.description = description
        self.llm = LLMService()
        self.assignment = AssignmentService()
        self.log = get_logger(__name__)
        setup_logging()
    
    @abstractmethod
    def solve(self):
        pass

