from fasttext import load_model, train_supervised
from typing import Optional

def quantize_model(path: str, output_location: str = "model_quantized.bin"):
    print("quantizing model...")
    model = load_model(path)

    model.quantize("labeled.data", retrain=True, epoch=50, cutoff=10000)
    model.save_model(output_location)
    print(f"model saved as `{output_location}`")
    return model

    print("\nquantized model:")

    example_count, precision, recall = model.test("valid.data")
    print(f"example count: {example_count}")
    print(f"precision: {round(precision, 2)*100}%")
    print(f"recall: {round(recall, 2)*100}%")

def train_model(input_data: str, output_location: str = "model_unquantized.bin"):
    model = train_supervised(input_data, lr=0.1, epoch=50)
    model.save_model(output_location)
    print(f"model saved as `{output_location}`")
    return model
    example_count, precision, recall = model.test("valid.data")
    print(f"example count: {example_count}")
    print(f"precision: {round(precision, 2)*100}%")
    print(f"recall: {round(recall, 2)*100}%\n")
    print(f"model saved as `{output_location}`")
