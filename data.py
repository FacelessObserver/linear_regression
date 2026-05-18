import torch as t
from pandas import DataFrame, read_csv
from typing import Callable

class Data:
    def __init__(self):
        self.data: DataFrame | None = None
    
    @staticmethod
    def sin_func(x):
        return t.sin(x)
    
    @staticmethod
    def galka_func(x):
        return 2**x * t.sin(2**-x)
    
    def _generate(self, x: t.Tensor, func: Callable, noise_level: float) -> None:
        y = func(x)
        y += t.randn(y.shape) * noise_level

        self.data = DataFrame({
            "x": x.numpy(),
            "y": y.numpy()
        })
    
    def gen_train_data(self, size: int, noise_level: float,
                       func: Callable, x_min: float, x_max: float) -> None:
        x = t.rand(size) * (x_max - x_min) + x_min
        self._generate(x, func, noise_level)

    def gen_test_data(self, size: int, noise_level: float,
                      func: Callable, x_min: float, x_max: float) -> None:
        x = t.linspace(x_min, x_max, size)
        self._generate(x, func, noise_level)

    def save(self, filename: str) -> bool:
        if self.data is None:
            return False
        self.data.to_csv(filename, index = False)
        return True
    
    def load(self, file_path: str) -> bool:
        try:
            self.data = read_csv(file_path)
            return True
        except Exception:
            return False
