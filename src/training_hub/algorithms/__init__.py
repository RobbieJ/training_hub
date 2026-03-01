from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from training_hub.model_capabilities import resolve_backend_selection


class Algorithm(ABC):
    """Base class for all training algorithms."""
    
    @abstractmethod
    def train(self, **kwargs) -> Any:
        """Execute the training algorithm."""
        pass
    
    @abstractmethod
    def get_required_params(self) -> Dict[str, Type]:
        """Return dictionary of required parameter names and their types."""
        pass

    @abstractmethod
    def get_optional_params(self) -> Dict[str, Type]:
        """Return dictionary of optional parameter names and their types."""
        pass

class Backend(ABC):
    """Base class for all backend implementations."""
    
    @abstractmethod
    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute training with the given parameters."""
        pass


class AlgorithmRegistry:
    """Registry for algorithms and their available backends."""
    
    _algorithms: Dict[str, Type[Algorithm]] = {}
    _backends: Dict[str, Dict[str, Type[Backend]]] = {}
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: Type[Algorithm]):
        """Register an algorithm class."""
        cls._algorithms[name] = algorithm_class
        if name not in cls._backends:
            cls._backends[name] = {}
    
    @classmethod
    def register_backend(cls, algorithm_name: str, backend_name: str, backend_class: Type[Backend]):
        """Register a backend implementation for an algorithm."""
        if algorithm_name not in cls._algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' must be registered before registering backends")
        cls._backends[algorithm_name][backend_name] = backend_class
    
    @classmethod
    def get_algorithm(cls, name: str) -> Type[Algorithm]:
        """Get algorithm class by name."""
        if name not in cls._algorithms:
            raise ValueError(f"Algorithm '{name}' not found")
        return cls._algorithms[name]
    
    @classmethod
    def get_backend(cls, algorithm_name: str, backend_name: str) -> Type[Backend]:
        """Get backend class for an algorithm."""
        if algorithm_name not in cls._backends:
            raise ValueError(f"Algorithm '{algorithm_name}' not found")
        if backend_name not in cls._backends[algorithm_name]:
            raise ValueError(f"Backend '{backend_name}' not found for algorithm '{algorithm_name}'")
        return cls._backends[algorithm_name][backend_name]
    
    @classmethod
    def list_algorithms(cls) -> list[str]:
        """List all registered algorithms."""
        return list(cls._algorithms.keys())
    
    @classmethod
    def list_backends(cls, algorithm_name: str) -> list[str]:
        """List all backends for an algorithm."""
        if algorithm_name not in cls._backends:
            return []
        return list(cls._backends[algorithm_name].keys())


def create_algorithm(
    algorithm_name: str,
    backend_name: str = None,
    model_path_or_architecture: str | None = None,
    trust_remote_code: bool | None = None,
    **kwargs,
) -> Algorithm:
    """Factory function to create algorithm instances with specified backend."""
    algorithm_class = AlgorithmRegistry.get_algorithm(algorithm_name)

    available_backends = AlgorithmRegistry.list_backends(algorithm_name)
    backend_name = resolve_backend_selection(
        algorithm_name=algorithm_name,
        backend_name=backend_name,
        available_backends=available_backends,
        model_path_or_architecture=model_path_or_architecture,
        trust_remote_code=trust_remote_code,
    )

    backend_class = AlgorithmRegistry.get_backend(algorithm_name, backend_name)
    backend_instance = backend_class()

    return algorithm_class(backend=backend_instance, **kwargs)
