from ..models.layout_models.azure_layout import AzureLayout
from ..models.layout_models.surya_layout import SuryaLayout
from ..models.layout_models.azure_layout import AzureLayout

from surya.settings import settings
from ..utils.utils import draw_color_bboxes, precision_recall

from dotenv import load_dotenv

import glob

import json
import os
import datasets
import pdb


#NOTE Check that datasets are identical

def extract_number(s):
    return int(''.join(filter(str.isdigit, s)))

class LayoutEvaluation():
    def __init__(self, model_names, dataset_gt_name, dataset_root_dir, layout_mapping, max=None, metrics = ["precision", "recall"]):
        """
        Args:
            model_names (list): List of model names ('Azure', 'Surya')
            dataset_gt_name (str): Name of the dataset to use for the ground truth ('Publaynet')
            dataset_root_dir (str): Root directory of the dataset.
            layout_mapping (dict): Mapping of layout types to the models. Use default. 
            max (int): Maximum number of documents to use. Use all by default.
            metrics (list): List of metrics to use.

        Return:
        layout_results_dict = {
            "document_idx": {
                "Figures": {page_num: [{"coordinates": [x1, y1, x2, y2],
                                                "span_pages": [int]}}],
                "Tables": {page_num: [{"coordinates": [x1, y1, x2, y2],
                                                "span_pages": [int]}}],
                "Text": {page_num: [{"coordinates": [x1, y1, x2, y2],
                                                "span_pages": [int]}}],
                "Titles": {page_num: [{"coordinates": [x1, y1, x2, y2],
                                                "span_pages": [int]}}]
                }
            }
        """
        self.model_names = model_names
        self.models = {}
        self.dataset_gt_name = dataset_gt_name
        self.dataset_root_dir = dataset_root_dir
        self.max = max
        self.metrics = metrics
        self.layout_mapping = layout_mapping
        self.dataset_gt, self.dataset_images =self.load_dataset()
        self.base_dir = os.path.join(os.getcwd().split("evals")[0], "evals/ocr_eval")
        self.eval_results = {}
        if self.dataset_gt_name == "publaynet":
            self.documents = glob.glob(os.path.join(self.dataset_root_dir, "*.pdf"))
            self.documents = sorted(self.documents, key=lambda x: extract_number(x))[0:self.max] 
        else:
            raise ValueError(f"Dataset {self.dataset_gt_name} not supported")

    def prepare_models(self, ocr_result_paths = ["data/ocr_results/azure_results.pkl", "data/ocr_results/surya_results.pkl"]):
        for i, model_name in enumerate(self.model_names):
            if model_name == "Azure":
                load_dotenv(os.path.join(self.base_dir, '.env'))
                endpoint = str(os.getenv('AZURE_API_URL'))
                key = str(os.getenv('AZURE_API_KEY'))
                print("Loading Azure Document Intelligence Layout Model")
                azure_layout = AzureLayout(model_name="Azure", evals=["layout"], layout_mapping=self.layout_mapping, results_path=os.path.join(self.base_dir, ocr_result_paths[i]))
                azure_models = azure_layout.load_models(endpoint, key)
                print("Retrieving Azure OCR Results")
                azure_results = azure_layout.run_ocr(azure_models, document_paths=self.documents)
                print("Transform Azure OCR Results")
                azure_results = azure_layout.results_transform(eval="layout", results = azure_results)
                self.models[model_name] = {}
                self.models[model_name]['layout'] = azure_layout
                self.models[model_name]['models'] = azure_models
                self.models[model_name]['results'] = azure_results
            elif model_name == "Surya":
                print("Loading Surya Layout Model")
                surya_layout = SuryaLayout(model_name="Surya", evals=["layout"], layout_mapping=self.layout_mapping, results_path=os.path.join(self.base_dir, ocr_result_paths[i]))
                surya_models = surya_layout.load_models()
                print("Retrieving Surya OCR Results")
                surya_results = surya_layout.run_ocr(surya_models, document_paths=self.documents)
                print("Transforming Surya OCR Results")
                surya_results = surya_layout.results_transform(eval="layout", results = surya_results)
                self.models[model_name] = {}
                self.models[model_name]['layout'] = surya_layout
                self.models[model_name]['models'] = surya_models
                self.models[model_name]['results'] = surya_results
            else:
                raise ValueError(f"Model {model_name} not supported")

    def run_eval(self):
        #pdb.set_trace()
        running_average = {}
        for model_name in self.model_names:
            self.eval_results[model_name + "_average"] = {}
            self.eval_results[model_name + "_average"]["count"] = {}
            for label in self.layout_mapping.keys():
                self.eval_results[model_name + "_average"]["count"][label] = 0
            for metric in self.metrics:
                self.eval_results[model_name + "_average"][metric] = {}
                for label in self.layout_mapping.keys():
                    self.eval_results[model_name + "_average"][metric][label] = 0
                self.eval_results[model_name + "_average"][metric]["total"] = 0

        for model_name in self.model_names:
            running_average[model_name] = {}
            self.eval_results[model_name] = {}
            print("Running evaluation on Model {}".format(model_name))
            for label in self.layout_mapping.keys():
                running_average[model_name][label] = {}
                for metric in self.metrics:
                    running_average[model_name][label][metric] = {'total metric': 0, 'total bboxes': 0}
            for idx, label_document in self.dataset_gt.items():
                self.eval_results[model_name][idx] = {}
                for label, document in label_document.items():
                    self.eval_results[model_name][idx][label] = {}
                    for page, bboxes in document.items():
                        self.eval_results[model_name][idx][label][page] = {}
                        try:
                            model_bboxes = [self.models[model_name]['results'][idx][label][page][i]['coordinates'] for i in range(len(self.models[model_name]['results'][idx][label][page]))]
                        except:
                            raise Exception("Model {} did not return a result for document {} page {}".format(model_name, idx, page))
                        gt_bboxes = [bbox['coordinates'] for bbox in bboxes]
                        for metric in self.metrics:
                            if metric in ["precision", "recall"]:
                                self.eval_results[model_name][idx][label][page][metric] = {}
                                metric_val = precision_recall(model_bboxes, gt_bboxes, penalize_double=False)
                                weight = len(gt_bboxes)
                                self.eval_results[model_name][idx][label][page][metric]['val'] = metric_val[metric]
                                self.eval_results[model_name][idx][label][page]['weight'] = weight
                                running_average[model_name][label][metric]['total metric'] += metric_val[metric] * weight
                                running_average[model_name][label][metric]['total bboxes'] += weight 
                            else:
                                raise ValueError(f"Metric {metric} not supported")
                #pdb.set_trace()
                for label in self.layout_mapping.keys():
                    for metric in self.metrics:
                        if running_average[model_name][label][metric]['total bboxes'] > 0:
                            self.eval_results[model_name + "_average"][metric][label] = running_average[model_name][label][metric]['total metric'] / running_average[model_name][label][metric]['total bboxes']
                        else:
                            self.eval_results[model_name + "_average"][metric][label] = 0
                        self.eval_results[model_name + "_average"]["count"][label] =  running_average[model_name][label][metric]['total bboxes']
                for metric in self.metrics:
                    if sum([running_average[model_name][label][metric]['total bboxes'] for label in self.layout_mapping.keys()]) > 0:
                        self.eval_results[model_name + "_average"][metric]["total"] = sum([running_average[model_name][label][metric]['total metric'] for label in self.layout_mapping.keys()]) / sum([running_average[model_name][label][metric]['total bboxes'] for label in self.layout_mapping.keys()])
                    else:
                        self.eval_results[model_name + "_average"][metric]["total"] = 0


    def load_dataset(self):
        if self.dataset_gt_name == "publaynet":
            dataset = datasets.load_dataset(settings.LAYOUT_BENCH_DATASET_NAME, split=f"train[:{self.max}]") 
            transformed_dataset = {}
            transformed_dataset_images = {}
            for idx, document in enumerate(dataset):
                transformed_dataset[idx] = {}
                transformed_dataset_images[idx] = {1: document['image'].convert("RGB")}
                for label in self.layout_mapping.keys():
                    correct_labels = self.layout_mapping[label][self.dataset_gt_name]
                    transformed_dataset[idx][label] = {1: [{'coordinates': bbox, 'span_pages': [1]} for bbox, label in zip(document["bboxes"], document["labels"]) if label in correct_labels]}
            return transformed_dataset, transformed_dataset_images
        else:
            raise ValueError(f"Dataset {self.dataset_gt_name} not supported")
    

    def visualize_eval(self, dir_path=None):
        if dir_path is None:
            dir_path = os.path.join(self.base_dir, "results/benchmark/layout_bench")
        
        label_colors = {
            "Figures": "blue",
            "Tables": "green",
            "Text": "red",
            "Titles": "purple"
        }
        
        for idx, label_document in self.dataset_gt.items():
            for page in range(1, len(label_document['Text']) + 1):
                image = self.dataset_images[idx][page].copy()
                for model_name in self.model_names + ["gt"]:
                    model_image = image.copy()
                    for label_col in self.layout_mapping:
                        if model_name == "gt":
                            bboxes_gt = [bbox['coordinates'] for bbox in label_document[label_col][page]]
                            model_image = draw_color_bboxes(bboxes_gt, model_image, color=label_colors[label_col]) 
                        else:
                            model_bboxes = [self.models[model_name]['results'][idx][label_col][page][i]['coordinates'] for i in range(len(self.models[model_name]['results'][idx][label_col][page]))]
                            model_image = draw_color_bboxes(model_bboxes, model_image, color=label_colors[label_col])
                    
                    model_image.save(os.path.join(dir_path, "{}_{}_{}_{}.png".format(idx, page, model_name, "_".join(label_colors.keys()))))

