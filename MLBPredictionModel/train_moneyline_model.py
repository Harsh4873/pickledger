from moneyline_model import train_moneyline_model


if __name__ == "__main__":
    result = train_moneyline_model()
    metadata = result["metadata"]
    print("Training rows:", metadata["training_rows"])
    print("Validation rows:", metadata["validation_rows"])
    print("Model metrics:", metadata["model_metrics"])
    print("Heuristic metrics:", metadata["heuristic_metrics"])
