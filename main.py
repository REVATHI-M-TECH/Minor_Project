from src.train import train
from src.evaluate import evaluate_model

if __name__ == "__main__":

    model, history, test_data = train()

    evaluate_model(model, history, test_data)
