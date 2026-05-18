import os
from utils.data import Data
from net.net import SinNet
from net.train import Trainer
from net.test import Tester
from utils.drawer import Drawer  
from config import *

def main():
    os.makedirs("datasets", exist_ok = True)
    os.makedirs("models", exist_ok = True)

    train_data = Data()
    if os.path.exists(TRAIN_PATH):
        print(f"Загрузка тренировочных данных из {TRAIN_PATH}")
        train_data.load(TRAIN_PATH)
    else:
        print(f"Создние тренировочных данных")
        train_data.gen_train_data(
            size = TRAIN_SIZE, noise_level = TRAIN_NOISE_LEVEL,
            func = FUNC, x_min = X_MIN, x_max = X_MAX
        )
        train_data.save(TRAIN_PATH)

    test_data = Data()
    if os.path.exists(TEST_PATH):
        print(f"Загрузка тестовых данных из {TEST_PATH}")
        test_data.load(TEST_PATH)
    else:
        print(f"Создание тестовых данных")
        test_data.gen_test_data(
            size = TEST_SIZE, noise_level = TEST_NOISE_LEVEL,
            func = FUNC, x_min = X_MIN, x_max = X_MAX
        )
        test_data.save(TEST_PATH)

    model = SinNet(hidden_neurons = HIDDEN_NEURONS)
    print(f"Модель создана с {HIDDEN_NEURONS} нейронами в скрытом слое")
    
    print("Производится обучение")
    trainer = Trainer(train_data.data, model)
    losses = trainer.train(
        epochs = EPOCHS,
        lr = LEARNING_RATE,
        batch_size = BATCH_SIZE,
        criterion = CRITERION,
        optimizer_class = OPTIMIZER,
        **OPTIMIZER_KWARGS
    )

    trainer.save(MODEL_PATH)

    Drawer.plot_loss(losses)

    print("Производится тестирование")
    tester = Tester(test_data.data, model)
    metrics = tester.test(batch_size = BATCH_SIZE, criterion = CRITERION)
    
    print(f"\nИтоговые метрики:")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"Max Error: {metrics['max_error']:.6f}")
    
    Drawer.plot_results(
        x_values = test_data.data["x"].values,
        predictions = metrics["predictions"],
        targets = metrics["targets"]
    )

if __name__ == "__main__":
    main()
