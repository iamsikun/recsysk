from __future__ import annotations


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def register(self, name: str | None = None):
        """
        A decorator to register a class in the registry.
        Usage: @Registry.register('name')
        Args:
            name: The name of the class to register.
        Returns:
            The decorated class.
        """

        def _register(cls):
            # use the class name if no specific name is provided
            key = name if name is not None else cls.__name__.lower()
            if key in self._registry:
                raise ValueError(
                    f"Class {cls.__name__} already registered with name {key}"
                )
            self._registry[key] = cls
            return cls

        return _register

    def get(self, name: str):
        """
        Retrieve a class by name.
        Args:
            name: The name of the class to retrieve.
        Returns:
            The class.
        """
        if name not in self._registry:
            raise KeyError(
                f"Class {name} not found in {self._name} registry. Available classes: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def build(self, config, **kwargs):
        """
        Instantiates the class using a config distionary.
        Assumes the config dictionary has a 'name' key and other keys are arguments
        **kwargs allows injecting runtime dependencies like feature_map, etc.
        """
        # make a deep copy to avoid modifying the original config
        args = config.copy()
        name = args.pop("name")  # remove name to pass the rest as kwargs
        cls = self.get(name)

        # merge config args with kwargs
        args.update(kwargs)

        return cls(**args)


DATASET_REGISTRY = Registry("Datasets")
MODEL_REGISTRY = Registry("Models")
OPTIMIZER_REGISTRY = Registry("Optimizers")
LOSS_REGISTRY = Registry("Losses")
