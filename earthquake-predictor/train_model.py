from model_utils import train_and_save_model


if __name__ == "__main__":
    metrics = train_and_save_model()
    print("Earthquake magnitude model trained")
    print(f"Records: {metrics['records']}")
    print(f"Mean absolute error: {metrics['mae']:.3f}")
    print(f"R2 score: {metrics['r2']:.3f}")
