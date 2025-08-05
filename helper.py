#!/usr/bin/env -S uv run --script

from click import option, group, echo, confirm, Path as ClickPath, prompt
from src.train import quantize_model, train_model
from floret import load_model
from src.format import format_data
from typing import Optional
from os import path

PathType = ClickPath()

@group()
def cli():
    pass

def check_options(input: Optional[str] = None, output: Optional[str] = None):
    if input and not path.exists(input):
        echo(f"input file `{input}` does not exist.")
        return False

    if output and path.exists(output) and not confirm(f"output file `{output}` already exists. overwrite?"):
        echo("not overwriting")
        return False

    return True

@cli.command("quantize", help="quantize the model to produce a smaller model.")
@option("--input", type=PathType, default="model_unquantized.bin", help="path to the model to be quantized")
@option("--output", type=PathType, default="model_quantized.bin", help="output path of quantized model")
@option("--cutoff", type=int, default=0, help="cutoff value for quantization")
def quantize(input: str, output: str, cutoff: int):
    if not check_options(input, output):
        return

    model = quantize_model(input, output, cutoff)

    if not confirm("test model?"): return
    example_count, precision, recall = model.test("valid.data")
    echo(f"example count: {example_count}")
    echo(f"precision: {round(precision, 2)*100}%")
    echo(f"recall: {round(recall, 2)*100}%\n")

@cli.command("train", help="train the model on a dataset.")
@option("--input", type=PathType, default="labeled.data", help="path to the dataset to be trained")
@option("--output", type=PathType, default="model_unquantized.bin", help="output path of trained model")
def train(input: str, output: str):
    if not check_options(input, output):
        return

    model = train_model(input, output)
    if not confirm("test model?"): return
    example_count, precision, recall = model.test("valid.data")
    echo(f"example count: {example_count}")
    echo(f"precision: {round(precision, 2)*100}%")
    echo(f"recall: {round(recall, 2)*100}%\n")

@cli.command("format", help="format dataset to be used for training.")
@option("--input", type=PathType, default="train.csv", help="path to the dataset to be formatted")
@option("--output", type=PathType, default="labeled.data", help="path to the formatted dataset")
def format(input: str, output: str):
    if not check_options(input, output):
        return
    format_data(input, output)

@cli.command("test", help="test the model yourself")
@option("--model", "model_path", type=PathType, default="model_unquantized.bin", help="path to the model to be tested")
def test(model_path: str):
    if not check_options(model_path):
        return

    model = load_model(model_path)
    echo("type `<exit>` or use Ctrl+C to exit")
    while True:
        text = prompt(">", prompt_suffix=" ")
        if text == "<exit>":
            break
        label, value = model.predict(str(text))
        echo(f"{label[0].replace('__label__','')}: {round(value[0], 2)*100}%")

if __name__ == "__main__":
    cli()
