from fasttext import load_model, train_supervised
from click import command, option

def quantize_model():
    print("quantizing model...")
    model = load_model("model_unqiantized.bin")

    model.quantize("labeled.data", retrain=True, epoch=50, cutoff=10000)
    model.save_model("model_quantized.bin")

    print("\nquantized Model:")
    example_count, precision, recall = model.test("valid.data")
    print(f"example count: {example_count}")
    print(f"precision: {round(precision, 2)*100}%")
    print(f"recall: {round(recall, 2)*100}%")

@command()
@option("--quantize", is_flag=True, help="quantize the model after training")
@option("--no-train", is_flag=True, help="skip training the model")
def main(quantize: bool, no_train: bool):
    if not no_train:
        model = train_supervised("labeled.data", lr=0.1, epoch=50)
        model.save_model("model_unquantized.bin")
        print("\noriginal Model:")
        example_count, precision, recall = model.test("valid.data")
        print(f"example count: {example_count}")
        print(f"precision: {round(precision, 2)*100}%")
        print(f"recall: {round(recall, 2)*100}%\n")
        print("model saved as `model_unquantized.bin`")
    else:
        print("skipping training...")
    if quantize:
        quantize_model()

if __name__ == '__main__':
    main()