import abc
from torch.utils.data import DataLoader, Dataset

class TrainingSessionInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fit') and 
                callable(subclass.fit) and 
                (hasattr(subclass, 'test') and 
                callable(subclass.test) and 
                hasattr(subclass, 'save_model') and 
                callable(subclass.save_model)  and 
                hasattr(subclass, 'load_model') and 
                callable(subclass.load_model) or 
                NotImplemented))
        
    @abc.abstractmethod
    def __init__(self, hyperparams: dict):
        """
        Enforces a specific constructor signature.
        """
        pass
    
    @abc.abstractmethod 
    def fit(self, train_dt, base_replay_dt=None) -> None:
        """If starting_checkpoint_path is not None, the model should be loaded from the checkpoint"""
        """Else, the model to be trained is considered to be the base model."""
        raise NotImplementedError

    @abc.abstractmethod 
    def test(self, test_dt: Dataset) -> float:
        """Test the trained model on the test data, must return the test accuracy"""
        raise NotImplementedError
        
    @abc.abstractmethod
    def save_model(self, full_file_path: str) -> None:
        """Save the model to the given path"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def init_model(self, full_file_path: str) -> None:
        """Load the model from the given path"""
        raise NotImplementedError
    