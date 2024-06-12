import os
import datasets
from surya.settings import settings

def load_huggingface_data(dataset_name, dir, max, langs = ['en']):
    base_dir = os.path.join(os.getcwd().split("OCR_Eval")[0], "OCR_Eval/ocr_eval")
    os.makedirs(os.path.join(base_dir, "data", dir), exist_ok=True)
    if max is None:
        dataset = datasets.load_dataset(dataset_name, split=f"train")
    else:
        dataset = datasets.load_dataset(dataset_name, split=f"train[:{max}]")
    if "language" in dataset.features.keys():
        dataset = dataset.filter(lambda x: x["language"] in langs)
    images = list(dataset["image"])
    images = [i.convert("RGB") for i in images]
    for index, img in enumerate(images):
        img.save(os.path.join(base_dir, "data", dir, f"{index}.pdf"), resolution=300, quality=95)

if __name__ == "__main__":
    layout_bench_dataset_name = settings.LAYOUT_BENCH_DATASET_NAME
    text_extraction_dataset_name = settings.RECOGNITION_BENCH_DATASET_NAME
    name_dir = "layout_bench/publaynet"
    max = None
    load_huggingface_data(layout_bench_dataset_name, name_dir, max)