from ..models.text_extraction_models.azure_text_extraction import AzureTextExtraction
from ..models.text_extraction_models.surya_text_extraction import SuryaTextExtraction
from ..models.text_extraction_models.tesseract_text_extraction import TesseractTextExtraction
from ..utils.utils import *

from surya.settings import settings
from surya.benchmark.metrics import precision_recall

from dotenv import load_dotenv

import glob

import os
import datasets
import pdb
import string
#NOTE move metrics functions here

#NOTE Check that datasets are identical

def extract_number(s):
    return int(''.join(filter(str.isdigit, s)))

class TextExtractionEvaluation():
    def __init__(self, model_names, dataset_gt_name, dataset_root_dir, max=None, metrics = ["text similarity"]):
        """
        Args:
            model_names (list): List of model names ('Azure', 'Surya')
            dataset_gt_name (str): Name of the dataset to use for the ground truth ('Publaynet')
            dataset_root_dir (str): Root directory of the dataset.
            text_extraction_mapping (dict): Mapping of text_extraction types to the models. Use default. 
            max (int): Maximum number of documents to use. Use all by default.
            metrics (list): List of metrics to use.

        Return:
        format:
        {idx: {
            page_num: {
                "text": text,
                "bboxes": [{"coordinates": bbox, "text": text}]
            }
        }}

        """
        self.model_names = model_names
        self.models = {}
        self.dataset_gt_name = dataset_gt_name
        self.dataset_root_dir = dataset_root_dir
        self.max = max
        self.metrics = metrics
        print("loading dataset")
        self.dataset_gt, self.dataset_gt_images = self.load_dataset()
        self.base_dir = os.path.join(os.getcwd().split("OCR_Eval")[0], "OCR_Eval/ocr_eval")
        self.eval_results = {}
        if self.dataset_gt_name == "vik_text_extraction_bench":
            self.documents = glob.glob(os.path.join(self.dataset_root_dir, "*.pdf"))
            self.documents = sorted(self.documents, key=lambda x: extract_number(x))[0:self.max] 
        else:
            raise ValueError(f"Dataset {self.dataset_gt_name} not supported")

    def prepare_models(self, ocr_result_paths = ["data/ocr_results/azure_results.pkl", "data/ocr_results/surya_text_extraction_results.pkl"]):
        for i, model_name in enumerate(self.model_names):
            if model_name == "Azure":
                load_dotenv(os.path.join(self.base_dir, '.env'))
                endpoint = str(os.getenv('AZURE_API_URL'))
                key = str(os.getenv('AZURE_API_KEY'))
                print("Loading Azure Document Intelligence text_extraction Model")
                azure_text_extraction = AzureTextExtraction(model_name="Azure", evals=["text"], results_path=os.path.join(self.base_dir, ocr_result_paths[i]))
                azure_models = azure_text_extraction.load_models(endpoint, key)
                print("Retrieving Azure OCR Results")
                azure_results = azure_text_extraction.run_ocr(azure_models, document_paths=self.documents)
                print("Transform Azure OCR Results")
                azure_results = azure_text_extraction.results_transform(eval="text", results = azure_results)
                self.models[model_name] = {}
                self.models[model_name]['text_extraction'] = azure_text_extraction
                self.models[model_name]['models'] = azure_models
                self.models[model_name]['results'] = azure_results
            elif model_name == "Surya":
                print("Loading Surya text_extraction Model")
                surya_text_extraction = SuryaTextExtraction(model_name="Surya", evals=["text"], results_path=os.path.join(self.base_dir, ocr_result_paths[i]))
                surya_models = surya_text_extraction.load_models()
                print("Retrieving Surya OCR Results")
                surya_results = surya_text_extraction.run_ocr(surya_models, document_paths=self.documents)
                print("Transforming Surya OCR Results")
                surya_results = surya_text_extraction.results_transform(eval="text", results = surya_results)
                self.models[model_name] = {}
                self.models[model_name]['text_extraction'] = surya_text_extraction
                self.models[model_name]['models'] = surya_models
                self.models[model_name]['results'] = surya_results
            elif model_name == "Tesseract":
                print("Loading Tesseract text_extraction Model")
                tesseract_text_extraction = TesseractTextExtraction(model_name="Tesseract", evals=["text"], results_path=os.path.join(self.base_dir, ocr_result_paths[i]))
                tesseract_models = tesseract_text_extraction.load_models()
                print("Retrieving Tesseract OCR Results")
                tesseract_results = tesseract_text_extraction.run_ocr(tesseract_models, document_paths=self.documents)
                print("Transforming Tesseract OCR Results")
                tesseract_results = tesseract_text_extraction.results_transform(eval="text", results=tesseract_results)
                self.models[model_name] = {}
                self.models[model_name]['text_extraction'] = tesseract_text_extraction
                self.models[model_name]['models'] = tesseract_models
                self.models[model_name]['results'] = tesseract_results
            else:
                raise ValueError(f"Model {model_name} not supported")

    def run_eval(self):
        running_average = {}
        for model_name in self.model_names:
            running_average[model_name] = {}
            self.eval_results[model_name] = {}
            # Currently only supports "text similarity"
            for metric in self.metrics:
                print("Running evaluation on Model {}. Metric: {}".format(model_name, metric))
                running_average[model_name][metric] = {}
                running_average[model_name][metric]['total metric'] = 0
                running_average[model_name][metric]['total word count'] = 0
                for idx, pages in self.dataset_gt.items():
                    if idx not in self.eval_results:
                        self.eval_results[idx] = {}
                    for page, content in pages.items():
                        if page not in self.eval_results[idx]:
                            self.eval_results[idx][page] =  {model_name: {}}
                        else:
                            self.eval_results[idx][page][model_name] = {}
                        if metric == "text similarity":
                            model_page_text = self.models[model_name]['results'][idx][page]['text']
                            gt_text = content['text']
                            metric_val = get_text_similarity(gt_text, model_page_text)
                            weight = len(gt_text)
                            self.eval_results[idx][page][model_name][metric] = metric_val
                            running_average[model_name][metric]['total metric'] += metric_val * weight
                            running_average[model_name][metric]['total word count'] += weight 
                        else:
                            raise ValueError(f"Metric {metric} not supported")
                if running_average[model_name][metric]['total word count'] > 0:
                    self.eval_results[model_name][metric] = running_average[model_name][metric]['total metric'] / running_average[model_name][metric]['total word count']
                    self.eval_results[model_name]["count"] = running_average[model_name][metric]['total word count']
                else:
                    self.eval_results[model_name][metric] = 0


    def load_dataset(self):
        if self.dataset_gt_name == "vik_text_extraction_bench":
            dataset = datasets.load_dataset(settings.RECOGNITION_BENCH_DATASET_NAME, split="train")
            langs = ['en']
            dataset = dataset.filter(lambda x: x["language"] in langs)
            if self.max is not None:
                dataset = dataset[:self.max]
            transformed_dataset = {}
            transformed_dataset_images = {}
            for idx, document in enumerate(dataset['text']):
                #this benchmark is only single page pdfs]
                transformed_dataset_images[idx] = dataset['image'][idx].convert("RGB")
                transformed_dataset[idx] = {1: {}}
                transformed_dataset[idx][1]['bboxes'] = []
                words = []
                for i, line in enumerate(document):
                    line_text = line.replace("\n", "")
                    for c in string.punctuation:
                        line_text = line_text.replace(c, "")
                    line_text_words = line_text.split(" ")
                    while "" in line_text_words:
                        line_text_words.remove("")
                    words += line_text_words
                    transformed_dataset[idx][1]['bboxes'].append({"coordinates": dataset["bboxes"][idx][i], "text": line_text})
                transformed_dataset[idx][1]['text'] = words
            return transformed_dataset, transformed_dataset_images
        else:
            raise ValueError(f"Dataset {self.dataset_gt_name} not supported")
        
    