import torch as t
from pandas import DataFrame
from torch.utils.data import DataLoader, TensorDataset
from .net import SinNet

class BaseModelHandler:
    def __init__(self, data: DataFrame, model: SinNet):
        self.data = data
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
    
    def _prepare_data(self, batch_size: int, shuffle: bool) -> DataLoader:
        x = t.tensor(self.data["x"].values, dtype = t.float32).reshape(-1, 1)
        y = t.tensor(self.data["y"].values, dtype = t.float32).reshape(-1, 1)

        return DataLoader(
            dataset = TensorDataset(x, y),
            batch_size = batch_size,
            shuffle = shuffle
        )
