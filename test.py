import numpy as np
import torch as t
from pandas import DataFrame
from .net import SinNet
from .handler import BaseModelHandler

class Tester(BaseModelHandler):
    def __init__(self, test_data: DataFrame, model: SinNet):
        super().__init__(test_data, model)
        self.model.eval()
    
    def test(self, batch_size: int, criterion: t.nn.Module) -> dict:

        dataloader = self._prepare_data(batch_size, shuffle = False)
        
        test_loss = 0
        predictions = []
        targets = []
        
        with t.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                y_pred = self.model(batch_x)
                
                predictions.append(y_pred.cpu())
                targets.append(batch_y.cpu())
                
                loss = criterion(y_pred, batch_y)
                test_loss += loss.item()
        
        predictions = t.cat(predictions).numpy().flatten()
        targets = t.cat(targets).numpy().flatten()

        avg_loss = test_loss / len(dataloader)
        mae = np.mean(np.abs(targets - predictions))
        max_error = np.max(np.abs(targets - predictions))
   
        return {
            "loss": avg_loss,
            "mae": mae,
            "max_error": max_error,
            "predictions": predictions,
            "targets": targets
        }
  