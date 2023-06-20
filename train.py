import fasttext

model = fasttext.train_supervised("all_categories.data", lr=0.5)
model.quantize("all_categories.data", retrain=True, cutoff=100000)
model.save_model("quantized_tagged.bin")

print(model.test("valid.data"))
