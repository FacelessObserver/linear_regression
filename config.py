from torch import nn, optim
from utils.data import Data

TRAIN_PATH = "datasets/train_data.csv"
TEST_PATH = "datasets/test_data.csv"
MODEL_PATH = "models/sin_model.pth"

TRAIN_SIZE = 100
TEST_SIZE = 100

TRAIN_NOISE_LEVEL = 0.2
TEST_NOISE_LEVEL = 0.0

X_MIN = -10
X_MAX = 10

FUNC = Data.galka_func

HIDDEN_NEURONS = 20
EPOCHS = 2000
LEARNING_RATE = 0.01
BATCH_SIZE = 100
CRITERION = nn.MSELoss()
OPTIMIZER = optim.Adam
OPTIMIZER_KWARGS = {}
