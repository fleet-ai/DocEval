from ..datasets.table_dataset import PDFTablesDataset
from ..datasets.table_dataset import get_structure_transform, get_detection_transform
from ..models.table_extraction_models.azure_table_extraction import AzureTableExtraction
from ..models.table_extraction_models.table_transformer_extraction import TableTransformerExtraction

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
from ocr_eval.utils.grits import grits_con, grits_top, grits_loc
from ocr_eval.utils import postprocess
from ocr_eval.utils import grits
import numpy as np


base_dir = os.path.join(os.getcwd().split("OCR_Eval")[0], "OCR_Eval/ocr_eval")

def flatten_tensor(tensor):
    flattened = tensor.reshape(-1, tensor.shape[-1])
    return flattened.tolist()

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

def objects_to_cells(bboxes, labels, scores, page_tokens, structure_class_names, structure_class_thresholds, structure_class_map):
    bboxes, scores, labels = postprocess.apply_class_thresholds(bboxes, labels, scores,
                                            structure_class_names,
                                            structure_class_thresholds)

    table_objects = []
    for bbox, score, label in zip(bboxes, scores, labels):
        table_objects.append({'bbox': bbox, 'score': score, 'label': label})
        
    table = {'objects': table_objects, 'page_num': 0} 
    
    table_class_objects = [obj for obj in table_objects if obj['label'] == structure_class_map['table']]
    if len(table_class_objects) > 1:
        table_class_objects = sorted(table_class_objects, key=lambda x: x['score'], reverse=True)
    try:
        table_bbox = list(table_class_objects[0]['bbox'])
    except:
        table_bbox = (0,0,1000,1000)
    
    tokens_in_table = [token for token in page_tokens if postprocess.iob(token['bbox'], table_bbox) >= 0.5]
    
    # Determine the table cell structure from the objects
    table_structures, cells, confidence_score = postprocess.objects_to_cells(table, table_objects, tokens_in_table,
                                                                    structure_class_names,
                                                                    structure_class_thresholds)
    
    return table_structures, cells, confidence_score

structure_class_names = [
    'table', 'table column', 'table row', 'table column header',
    'table projected row header', 'table spanning cell', 'no object'
]
structure_class_map = {k: v for v, k in enumerate(structure_class_names)}
structure_class_thresholds = {
    "table": 0.5,
    "table column": 0.5,
    "table row": 0.5,
    "table column header": 0.5,
    "table projected row header": 0.5,
    "table spanning cell": 0.5,
    "no object": 10
}

class TableExtractionEvaluation:
    def __init__(self, model_names,  metrics, dataset_gt_name, test_max_size, batch_size, num_workers, eval_pool_size, results_path, debug = True, debug_save_dir=os.path.join(base_dir, 'results/benchmark/table_extraction')):
        self.model_names = model_names
        self.metrics = metrics
        self.models = {}
        self.dataset_gt_name = dataset_gt_name
        self.data_type = "structure"
        if dataset_gt_name == "fintabnet":
            self.dataset_root = os.path.join(base_dir, "data/table_extraction_bench/fintabnet/fintabnet_processed/FinTabNet.c_Image_Structure_PASCAL_VOC")
            self.table_words_dir = os.path.join(base_dir, 'data/table_extraction_bench/fintabnet/fintabnet_processed/FinTabNet.c_Image_Table_Words_JSON')
        else:
            raise ValueError("Invalid dataset_gt_name")
        if "Table_Transformer" in self.model_names:
            self.config_file = os.path.join(base_dir, "table-transformer/src/structure_config.json")
            self.model_load_path = os.path.join(base_dir, "model_weights/table-transformer/extraction_fintabnet.pth")
        self.test_max_size = test_max_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_pool_size = eval_pool_size
        self.eval_results = {}
        self.results_path = results_path
        self.debug = debug
        self.debug_save_dir = debug_save_dir

    def prepare_models(self):
        for i, model_name in enumerate(self.model_names):
            self.models[model_name] = {}
            if model_name == "Table_Transformer":
                print("initialize table transformer")
                results_path = self.results_path[i]
                table_transformer = TableTransformerExtraction(
                    self.dataset_root, self.table_words_dir, self.data_type, self.config_file,
                    self.test_max_size, self.batch_size, self.num_workers, self.eval_pool_size,
                    self.model_load_path, self.debug, results_path, self.debug_save_dir)
                print("get data")
                data_loader_test, dataset_test = table_transformer.get_data()
                print("load models")
                model, criterion, postprocessors = table_transformer.load_models()
                print("run ocr")
                ocr_results = table_transformer.run_ocr(model, data_loader_test)
                print("results transform")
                transformed_results = table_transformer.results_transform(ocr_results)
                self.models[model_name]['results'] = transformed_results
            elif model_name == "Azure":
                results_path = self.results_path[i]
                table_azure = AzureTableExtraction(
                    self.dataset_root, self.table_words_dir, self.data_type, self.config_file,
                    self.test_max_size, self.batch_size, self.num_workers, self.eval_pool_size, self.debug, results_path, self.debug_save_dir)
                print("get data")
                data_loader_test, dataset_test = table_azure.get_data()
                print("load models ")
                load_dotenv(os.path.join(base_dir, '.env'))
                endpoint = str(os.getenv('AZURE_API_URL'))
                key = str(os.getenv('AZURE_API_KEY'))
                models = table_azure.load_models(endpoint, key)
                print("run ocr")
                ocr_results = table_azure.run_ocr(models, data_loader_test)
                self.models[model_name]['raw_results'] = ocr_results
                print("results transform")
                transformed_results = table_azure.results_transform(ocr_results)
                self.models[model_name]['results'] = transformed_results
            else:
                raise ValueError("Model not currently supported by evaluation framework.")
    

    def run_eval(self):
        data_loader_test, dataset_test = self.get_data()
        self.models['ground_truth'] = []

        metrics_dict_total = {}
        metrics_expansion = {
            "grits_top": ["grits_top", "grits_precision_top", "grits_recall_top", "grits_top_upper_bound"],
            "grits_loc": ["grits_loc", "grits_precision_loc", "grits_recall_loc", "grits_loc_upper_bound"],
            "grits_con": ["grits_con", "grits_precision_con", "grits_recall_con", "grits_con_upper_bound"]
        }
        for model_name in self.model_names:
            self.eval_results[model_name + "_average"] = {}
            expanded_metrics = []
            for metric in self.metrics:
                expanded_metrics.extend(metrics_expansion[metric])
            metrics_dict_total[model_name] = {metric: 0 for metric in expanded_metrics}
            metrics_dict_total[model_name]['weight'] = 0

        i = 0
        for samples, targets in tqdm(data_loader_test, total=len(data_loader_test), desc="Processing", unit="batch"):
            for target in targets:
                image_id = target['img_path'].split('/')[-1]
                self.eval_results[image_id] = {}
                for model_name in self.model_names:
                    model_pred_boxes = self.models[model_name]['results'][i]['pred_bbox_grid']
                    model_pred_relspan = self.models[model_name]['results'][i]['pred_relspan_grid']
                    model_pred_text = self.models[model_name]['results'][i]['pred_text_grid']
                    if len(model_pred_boxes) == 0:
                        print("Skipping image: ", image_id, " for model: ", model_name)
                        self.eval_results[image_id][model_name] = "Skipped"
                        continue
                    true_img_size = list(reversed(target['orig_size'].tolist()))
                    true_bboxes = target['boxes']
                    true_bboxes = [elem.tolist() for elem in rescale_bboxes(true_bboxes, true_img_size)]
                    true_labels = target['labels'].tolist()
                    true_scores = [1 for elem in true_labels]
                    img_words_filepath = os.path.join(self.table_words_dir, image_id.replace(".jpg", "_words.json"))
                    with open(img_words_filepath, 'r') as f:
                        true_page_tokens = json.load(f)
                    true_table_structures, true_cells, _ = objects_to_cells(true_bboxes, true_labels, true_scores,
                                                                            true_page_tokens, structure_class_names,
                                                                            structure_class_thresholds, structure_class_map)
                    true_relspan_grid = np.array(grits.cells_to_relspan_grid(true_cells))
                    true_bbox_grid = np.array(grits.cells_to_grid(true_cells, key='bbox'))
                    true_text_grid = np.array(grits.cells_to_grid(true_cells, key='cell_text'), dtype=object)
                    self.models['ground_truth'].append({"true_relspan_grid": true_relspan_grid, "true_bbox_grid": true_bbox_grid, "true_text_grid": true_text_grid, "image_path": image_id})
                    if self.debug:
                        img_path = target['img_path']
                        bboxes_lst = flatten_tensor(true_bbox_grid)
                        img = Image.open(img_path)
                        img = draw_color_bboxes(bboxes_lst, img)
                        img_id = img_path.split("/")[-1].replace(".jpg", "_bboxes.jpg")
                        bboxes_out_filename = "gt_" + img_id
                        bboxes_out_filepath = os.path.join(self.debug_save_dir, bboxes_out_filename)
                        img.save(bboxes_out_filepath)
                    metrics_dict = {}
                    for metric in self.metrics:
                        if metric == 'grits_top':
                            (metrics_dict['grits_top'],
                                metrics_dict['grits_precision_top'],
                                metrics_dict['grits_recall_top'],
                                metrics_dict['grits_top_upper_bound']) = grits_top(true_relspan_grid,
                                                                                    model_pred_relspan)
                        elif metric == 'grits_loc':
                            (metrics_dict['grits_loc'],
                                metrics_dict['grits_precision_loc'],
                                metrics_dict['grits_recall_loc'],
                                metrics_dict['grits_loc_upper_bound']) = grits_loc(true_bbox_grid,
                                                                                    model_pred_boxes)
                        elif metric == 'grits_con':
                            (metrics_dict['grits_con'],
                                metrics_dict['grits_precision_con'],
                                metrics_dict['grits_recall_con'],
                                metrics_dict['grits_con_upper_bound']) = grits_con(true_text_grid,
                                                                                    model_pred_text)
                        else:
                            raise ValueError(f"Metric {metric} not supported")
                    self.eval_results[image_id][model_name] = metrics_dict
                    for key, val in metrics_dict.items():
                        metrics_dict_total[model_name][key] += val
                    metrics_dict_total[model_name]['weight'] += 1
            i += 1
        metric_dicts_average = {}
        for model_name in self.model_names:
            metric_dicts_average[model_name] = {}
            for key, val in metrics_dict_total[model_name].items():
                metric_dicts_average[model_name][key] = val / metrics_dict_total[model_name]['weight']
            self.eval_results[model_name + "_average"] = metric_dicts_average[model_name]



    def get_data(self):
        if self.dataset_gt_name == "fintabnet":
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
        self.models['ground_truth'] = []

        metrics_dict_total = {}
        for model_name in self.model_names:
            self.eval_results[model_name] = {}
            metrics_dict_total[model_name] = {'grits_top': 0, 'grits_loc': 0, 'grits_con': 0, 'grits_precision_top': 0, 'grits_precision_loc': 0, 'grits_precision_con': 0, 'grits_recall_top': 0, 'grits_recall_loc': 0, 'grits_recall_con': 0, 'grits_top_upper_bound': 0, 'grits_loc_upper_bound': 0, 'grits_con_upper_bound': 0, 'weight': 0} 
        if self.eval_type == "table_extraction":
            i=0
            for samples, targets in tqdm(data_loader_test, total=len(data_loader_test), desc="Processing", unit="batch"):
                for target in targets:
                    image_id = target['img_path'].split('/')[-1]
                    self.eval_results[image_id] = {}
                    for model_name in self.model_names:
                        model_pred_boxes = self.models[model_name]['results'][i]['pred_bbox_grid']
                        model_pred_relspan = self.models[model_name]['results'][i]['pred_relspan_grid']
                        model_pred_text = self.models[model_name]['results'][i]['pred_text_grid']
                        if len(model_pred_boxes)==0:
                            print("Skipping image: ", image_id, " for model: ", model_name)
                            self.eval_results[image_id][model_name] = "Skipped"
                            continue
                        true_img_size = list(reversed(target['orig_size'].tolist()))
                        true_bboxes = target['boxes']
                        true_bboxes = [elem.tolist() for elem in rescale_bboxes(true_bboxes, true_img_size)]
                        true_labels = target['labels'].tolist()
                        true_scores = [1 for elem in true_labels]
                        img_words_filepath = os.path.join(self.table_words_dir, image_id.replace(".jpg", "_words.json"))
                        with open(img_words_filepath, 'r') as f:
                            true_page_tokens = json.load(f)
                        true_table_structures, true_cells, _ = objects_to_cells(true_bboxes, true_labels, true_scores,
                                                        true_page_tokens, structure_class_names,
                                                        structure_class_thresholds, structure_class_map)
                        true_relspan_grid = np.array(grits.cells_to_relspan_grid(true_cells))
                        true_bbox_grid = np.array(grits.cells_to_grid(true_cells, key='bbox'))
                        true_text_grid = np.array(grits.cells_to_grid(true_cells, key='cell_text'), dtype=object)
                        self.models['ground_truth'].append({"true_relspan_grid": true_relspan_grid, "true_bbox_grid": true_bbox_grid, "true_text_grid": true_text_grid, "image_path": image_id})
                        if self.debug:
                            img_path = target['img_path']
                            bboxes_lst = flatten_tensor(true_bbox_grid)
                            img = Image.open(img_path)
                            img = draw_color_bboxes(bboxes_lst, img)
                            img_id = img_path.split("/")[-1].replace(".jpg", "_bboxes.jpg")
                            bboxes_out_filename = "gt_" + img_id
                            bboxes_out_filepath = os.path.join(self.debug_save_dir, bboxes_out_filename)
                            img.save(bboxes_out_filepath)
                        metrics_dict = {}
                        try:
                            (metrics_dict['grits_top'],
                            metrics_dict['grits_precision_top'],
                            metrics_dict['grits_recall_top'],
                            metrics_dict['grits_top_upper_bound']) = grits_top(true_relspan_grid,
                                                                            model_pred_relspan)
                        except:
                            pdb.set_trace()
                        # Compute GriTS_Loc (location)
                        (metrics_dict['grits_loc'],
                        metrics_dict['grits_precision_loc'],
                        metrics_dict['grits_recall_loc'],
                        metrics_dict['grits_loc_upper_bound']) = grits_loc(true_bbox_grid,
                                                                            model_pred_boxes)
                        # Compute GriTS_Con (text content)
                        (metrics_dict['grits_con'],
                        metrics_dict['grits_precision_con'],
                        metrics_dict['grits_recall_con'],
                        metrics_dict['grits_con_upper_bound']) = grits_con(true_text_grid,
                                                                    model_pred_text)
                        self.eval_results[image_id][model_name] = metrics_dict
                        for key, val in metrics_dict.items():
                            metrics_dict_total[model_name][key] += val
                        metrics_dict_total[model_name]['weight'] += 1
                    i += 1
            metric_dicts_average = {}
            for model_name in self.model_names:
                metric_dicts_average[model_name] = {}
                for key, val in metrics_dict_total[model_name].items():
                    metric_dicts_average[model_name][key] = val / metrics_dict_total[model_name]['weight']
                self.eval_results[model_name] = metric_dicts_average[model_name]
        else:
            raise ValueError("Invalid eval_type")  
    """