import fasttext

model = fasttext.train_supervised("labeled.data", lr=0.5, epoch=50)
model.quantize("labeled.data", retrain=True, cutoff=100000)
model.save_model("model.bin")

print(model.test("valid.data"))
