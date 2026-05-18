import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class Drawer:
    matplotlib.rcParams["figure.figsize"] = (13.0, 5.0)

    @staticmethod
    def plot_loss(losses: list[float], save_path: str | None = None):
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Test Loss History")
        plt.grid(True, alpha = 0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_results(x_values: np.ndarray, predictions: np.ndarray, 
                    targets: np.ndarray, save_path: str | None = None):
        plt.figure()
        plt.plot(x_values, targets, "o", label = "Ground truth")
        plt.plot(x_values, predictions, "o", c = "red", label = "Prediction")
        
        plt.legend(loc = "upper left")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title("Final Prediction")
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
