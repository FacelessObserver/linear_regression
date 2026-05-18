import torch as t
from pandas import DataFrame
from .net import SinNet
from .handler import BaseModelHandler

class Trainer(BaseModelHandler):
    def __init__(self, train_data: DataFrame, model: SinNet):
        super().__init__(train_data, model)
        self.model.train()
    
    def train(self, epochs: int, lr: float, batch_size: int,
              criterion: t.nn.Module, optimizer_class: t.optim.Optimizer,
              **optimizer_kwargs) -> list[float]:
        
        dataloader = self._prepare_data(batch_size, shuffle = True)

        optimizer = optimizer_class(
            params = self.model.parameters(),
            lr = lr,
            **optimizer_kwargs
        )
        
        losses = []
        for _ in range(epochs):
            epoch_loss = 0
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(batch_x)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

        return losses

    def save(self, path: str) -> bool:
        try:
            t.save(self.model.state_dict(), path)
            return True
        except Exception:
            return False
