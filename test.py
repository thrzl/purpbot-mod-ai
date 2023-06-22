import fasttext

model = fasttext.load_model("model.bin")
example_count, precision, recall = model.test("valid.data")
print(f"Example Count: {example_count}")
print(f"Precision: {round(precision, 2)*100}%")
print(f"Recall: {round(recall, 2)*100}%")

while True:
    prediction = model.predict(input("> "))
    label, value = prediction
    print(f"{label[0].replace('__label__','')}: {round(value[0], 2)*100}%")
