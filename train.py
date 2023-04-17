import fasttext

model = fasttext.train_supervised('all_categories.data', lr=0.5)

model.quantize('all_categories.data', retrain=True)

model.save_model('quantized_tagged.bin')