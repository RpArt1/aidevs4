from abc import ABC
from abc import abstractmethod

class Assignment(ABC):
    def __init__(self, name: str, description: str ) : 
        self.name = name
        self.description = description
    
    @abstractmethod
    def solve(self):
        pass

