from ..datasets.table_dataset import PDFTablesDataset
from ..datasets.table_dataset import get_structure_transform, get_detection_transform
from ..models.table_extraction_models.table_transformer_detection import TableTransformerDetection
from ..models.table_extraction_models.azure_table_detection import AzureTableDetection
import torch
from torch.utils.data import DataLoader
from ..utils.utils import collate_fn
import pdb
import os
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import json

from ..utils.utils import draw_color_bboxes, precision_recall



base_dir = os.path.join(os.getcwd().split("DocEval")[0], "DocEval/doceval")
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def rescale_bboxes_no_conversion(out_bbox, size):
    img_w, img_h = size
    b = out_bbox * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def get_transform(data_type, image_set):
        if data_type == 'structure':
            return get_structure_transform(image_set)
        else:
            return get_detection_transform(image_set)
        
def get_class_map(data_type):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6
        }
    else:
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map



class TableDetectionEvaluation:
    def __init__(self, model_names, metrics, dataset_gt_name, dataset_root_dir, test_max_size, batch_size, num_workers, eval_pool_size, results_path, debug, debug_save_dir=os.path.join(base_dir, 'results/benchmark/table_detection')):
        self.model_names = model_names
        self.metrics = metrics
        self.models = {}
        self.dataset_gt_name = dataset_gt_name
        self.dataset_root_dir = dataset_root_dir
        self.data_type = "detection"
        if dataset_gt_name == "pubtables":

            self.dataset_root = os.path.join(dataset_root_dir, "PubTables-1M-Detection")
            self.table_words_dir = os.path.join(dataset_root_dir, "PubTables-1M-Detection/words")
        else:
            raise ValueError("Invalid dataset_gt_name")

        if "Table_Transformer" in self.model_names:
            self.config_file = os.path.join(base_dir, "table-transformer/src/detection_config.json")
            self.model_load_path = os.path.join(base_dir, "model_weights/table-transformer/detection_pubtables1m.pth")

        self.test_max_size = test_max_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_pool_size = eval_pool_size
        self.eval_results = {}
        self.debug = debug
        self.results_path = results_path
        self.debug_save_dir = debug_save_dir
    def prepare_models(self):
        for i, model_name in enumerate(self.model_names):
            self.models[model_name] = {}
            if model_name == "Table_Transformer":
                print("initialize table transformer")
                print(self.dataset_root)
                results_path = self.results_path[i]
                table_transformer_detection = TableTransformerDetection(
                    self.dataset_root, self.table_words_dir, self.data_type, self.config_file,
                    self.test_max_size, self.batch_size, self.num_workers, self.eval_pool_size,
                    self.model_load_path, self.debug, results_path, self.debug_save_dir
                )
                print("get data")
                data_loader_test, dataset_test = table_transformer_detection.get_data()
                print("load models")
                model, criterion, postprocessors = table_transformer_detection.load_models()
                print("run ocr")
                ocr_results = table_transformer_detection.run_ocr(model, data_loader_test)
                print("results transform")
                transformed_results = table_transformer_detection.results_transform(ocr_results)
                self.models[model_name]['results'] = transformed_results
            elif model_name == "Azure":
                results_path = self.results_path[i]
                print("initiaize Azure")
                print(self.dataset_root)
                table_azure = AzureTableDetection(
                    self.dataset_root, self.table_words_dir, self.data_type, self.config_file,
                    self.test_max_size, self.batch_size, self.num_workers, self.eval_pool_size, self.debug, results_path, self.debug_save_dir
                )
                print("get data")
                data_loader_test, dataset_test = table_azure.get_data()
                print("load models ")
                load_dotenv(os.path.join(base_dir, '.env'))
                endpoint = str(os.getenv('AZURE_API_URL'))
                key = str(os.getenv('AZURE_API_KEY'))
                models = table_azure.load_models(endpoint, key)
                print("run ocr")
                ocr_results = table_azure.run_ocr(models, data_loader_test)
                print("results transform")
                transformed_results = table_azure.results_transform(ocr_results)
                self.models[model_name]['results'] = transformed_results

    def run_eval(self):
        data_loader_test, dataset_test = self.get_data()
        for model_name in self.model_names:
            self.eval_results[model_name] = {}
        for model_name in self.model_names:
            running_average = {model_name: {metric: {"total metric": 0, "total weight": 0} for metric in self.metrics} for model_name in self.model_names}
            i = 0
            for samples, targets in tqdm(data_loader_test, total=len(data_loader_test), desc="Processing", unit="batch"):
                for target in targets:
                    image_id = target['img_path'].split('/')[-1]
                    if image_id not in self.eval_results:
                        self.eval_results[image_id] = {}
                    for model_name in self.model_names:
                        if model_name not in self.eval_results[image_id]:
                            self.eval_results[image_id][model_name] = {}
                        model_boxes = self.models[model_name]['results'][i]['bboxes']
                        target_boxes = target['boxes']
                        img_size = [int(target['orig_size'][1]), int(target['orig_size'][0])]
                        target_boxes = [elem.tolist() for elem in rescale_bboxes(target_boxes, img_size)]
                        if self.debug:
                            image_path = os.path.join(self.debug_save_dir, "gt_" + image_id)
                            img = Image.open(target['img_path'])
                            img = draw_color_bboxes(target_boxes, img)
                            img.save(image_path)
                        for metric in self.metrics:
                            if metric == "precision":
                                precision_recall_results = precision_recall(model_boxes, target_boxes, penalize_double=False)
                                metric_val = precision_recall_results['precision']
                                weight = len(target_boxes)
                                self.eval_results[image_id][model_name][metric] = {"val": metric_val, "weight": weight}
                                running_average[model_name][metric]["total metric"] += metric_val * weight
                                running_average[model_name][metric]["total weight"] += weight
                            elif metric == "recall":
                                precision_recall_results = precision_recall(model_boxes, target_boxes, penalize_double=False)
                                metric_val = precision_recall_results['recall']
                                weight = len(target_boxes)
                                self.eval_results[image_id][model_name][metric] = {"val": metric_val, "weight": weight}
                                running_average[model_name][metric]["total metric"] += metric_val * weight
                                running_average[model_name][metric]["total weight"] += weight
                            else:
                                raise ValueError(f"Metric {metric} not supported")
                    i += 1
        for model_name in self.model_names:
            for metric in self.metrics:
                if running_average[model_name][metric]["total weight"] > 0:
                    self.eval_results[model_name][f"average {metric}"] = running_average[model_name][metric]["total metric"] / running_average[model_name][metric]["total weight"]
                    self.eval_results[model_name][f"weight {metric}"] = running_average[model_name][metric]["total weight"]
                else:
                    self.eval_results[model_name][f"average {metric}"] = 0 
                    self.eval_results[model_name][f"weight {metric}"] = 0


    def get_data(self):
        if self.dataset_gt_name == "pubtables":
            class_map = get_class_map(self.data_type)
            print("class_map: ", class_map)
            dataset_test = PDFTablesDataset(os.path.join(self.dataset_root,
                                                        "test"),
                                            get_transform(self.data_type, "val"),
                                            do_crop=False,
                                            max_size=self.test_max_size,
                                            make_coco=True,
                                            include_eval=True,
                                            image_extension=".jpg",
                                            xml_fileset="test_filelist.txt",
                                            class_map=class_map)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

            data_loader_test = DataLoader(dataset_test,
                                        self.batch_size,
                                        sampler=sampler_test,
                                        drop_last=False,
                                        collate_fn=collate_fn,
                                        num_workers=self.num_workers)
            return data_loader_test, dataset_test
        else:
            raise ValueError(f"Dataset {self.dataset_gt_name} not supported")
    


    """
    def run_eval(self):
        data_loader_test, dataset_test = self.get_data()
        for model_name in self.model_names:
            self.eval_results[model_name] = {}
        running_total_precision = {model_name: 0 for model_name in self.model_names}
        running_total_weight = {model_name: 0 for model_name in self.model_names}
        running_total_recall = {model_name: 0 for model_name in self.model_names}
        i=0
        for samples, targets in tqdm(data_loader_test, total=len(data_loader_test), desc="Processing", unit="batch"):
            for target in targets:
                image_id = target['img_path'].split('/')[-1]
                self.eval_results[image_id] = {}
                for model_name in self.model_names:
                    self.eval_results[image_id][model_name] = {"precision": {}, "recall": {}}
                    #pdb.set_trace()
                    model_boxes = self.models[model_name]['results'][i]['bboxes']
                    target_boxes = target['boxes']
                    img_size = [int(target['orig_size'][1]), int(target['orig_size'][0])]
                    target_boxes = [elem.tolist() for elem in rescale_bboxes(target_boxes, img_size)]
                    if self.debug:
                        image_path = os.path.join(self.debug_save_dir, "gt_"+image_id)
                        img = Image.open(target['img_path'])
                        img = draw_color_bboxes(target_boxes, img)
                        img.save(image_path)
                    if metric == ["precision", "recall"]:
                        precision_recall_results = precision_recall(model_boxes, target_boxes, penalize_double=False)
                        precision = precision_recall_results['precision']
                        recall = precision_recall_results['recall']
                        #if precision < 1:
                            #pdb.set_trace()
                        weight =len(target_boxes)
                        self.eval_results[image_id][model_name]['precision']['val'] = precision
                        self.eval_results[image_id][model_name]['precision']['weight'] = weight
                        self.eval_results[image_id][model_name]['recall']['val'] = recall
                        self.eval_results[image_id][model_name]['recall']['weight'] = weight
                        running_total_precision[model_name] += precision * weight
                        running_total_recall[model_name] += recall * weight
                        running_total_weight[model_name] += weight
                    else:
                        for metric in self.metrics:
                            if metric == "precision":
                                precision_recall_results = precision_recall(model_boxes, target_boxes, penalize_double=False)
                                precision = precision_recall_results['precision']
                                weight =len(target_boxes)
                                self.eval_results[image_id][model_name]['precision']['val'] = precision
                                self.eval_results[image_id][model_name]['precision']['weight'] = weight
                                running_total_precision[model_name] += precision * weight
                                running_total_weight[model_name] += weight
                            elif metric == "recall":
                                precision_recall_results = precision_recall(model_boxes, target_boxes, penalize_double=False)
                                recall = precision_recall_results['recall']
                                weight =len(target_boxes)
                                self.eval_results[image_id][model_name]['recall']['val'] = recall
                                self.eval_results[image_id][model_name]['recall']['weight'] = weight
                                running_total_recall[model_name] += recall * weight
                                running_total_weight[model_name] += weight
                            else:
                                raise ValueError(f"Metric {metric} not supported")

                i += 1
        for model_name in self.model_names:
            if running_total_weight[model_name] > 0:
                self.eval_results[model_name]['average precision 50%'] = running_total_precision[model_name] / running_total_weight[model_name]  
                self.eval_results[model_name]['average recall 50%'] = running_total_recall[model_name] / running_total_weight[model_name]
    
    
    
    
    """

