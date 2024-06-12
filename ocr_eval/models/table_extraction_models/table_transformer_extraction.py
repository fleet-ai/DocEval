import torch
import argparse
from torch.utils.data import DataLoader
from ocr_eval.datasets.table_dataset import PDFTablesDataset, get_structure_transform, get_detection_transform
from ocr_eval.utils.utils import collate_fn
import os
import json
from tqdm import tqdm
import pdb
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from ocr_eval.utils.utils import draw_color_bboxes
from ocr_eval.utils import postprocess
from ocr_eval.utils import grits
import numpy as np

base_dir = os.path.join(os.getcwd().split("OCR_Eval")[0], "OCR_Eval")

import sys
sys.path.append(os.path.join(base_dir, "ocr_eval/table-transformer/detr"))
from models import build_model


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


class TableTransformerExtraction:
    def __init__(self, dataset_root, table_words_dir, data_type, config_file, test_max_size, batch_size, num_workers, eval_pool_size, model_load_path, debug, results_path, debug_save_dir=os.path.join(base_dir, 'ocr_eval/results/benchmark/table_extraction'), device = "cpu"):
        self.dataset_root = dataset_root
        self.table_words_dir = table_words_dir
        self.data_type = data_type
        self.config_file = config_file
        self.test_max_size = test_max_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_pool_size = eval_pool_size
        self.model_load_path = model_load_path
        self.device = device
        self.debug = debug
        self.results_path = results_path
        self.debug_save_dir = debug_save_dir
        self.args = self.create_args()

    def create_args(self):
        cmd_args = argparse.Namespace()
        cmd_args.data_root_dir = self.dataset_root
        cmd_args.config_file = self.config_file
        cmd_args.data_type = self.data_type
        cmd_args.model_load_path = self.model_load_path
        cmd_args.table_words_dir = self.table_words_dir
        cmd_args.batch_size = self.batch_size
        cmd_args.num_workers = self.num_workers
        cmd_args.test_max_size = self.test_max_size
        cmd_args.eval_pool_size = self.eval_pool_size
        cmd_args.device = self.device
        cmd_args.backbone = 'resnet18'  # Set the backbone architecture
        cmd_args.mode = "eval"

        config_args = json.load(open(cmd_args.config_file, 'rb'))
        for key, value in cmd_args.__dict__.items():
            if key not in config_args or value is not None:
                config_args[key] = value

        args = argparse.Namespace(**config_args)
        print(args.__dict__)
        return args

    def get_transform(self, image_set):
        if self.data_type == 'structure':
            return get_structure_transform(image_set)
        else:
            return get_detection_transform(image_set)

    def get_class_map(self):
        if self.data_type == 'structure':
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

    def get_data(self):
        class_map = self.get_class_map()
        dataset_test = PDFTablesDataset(
            os.path.join(self.dataset_root, "test"),
            self.get_transform("val"),
            do_crop=False,
            max_size=self.test_max_size,
            make_coco=True,
            include_eval=True,
            image_extension=".jpg",
            xml_fileset="test_filelist.txt",
            class_map=class_map
        )
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = DataLoader(
            dataset_test,
            self.batch_size,
            sampler=sampler_test,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers
        )
        return data_loader_test, dataset_test

    def load_models(self):
        model, criterion, postprocessors = build_model(self.args)
        model.to(self.device)
        if self.model_load_path:
            print("Loading model from checkpoint")
            loaded_state_dict = torch.load(self.model_load_path, map_location=self.device)
            model_state_dict = model.state_dict()
            pretrained_dict = {
                k: v
                for k, v in loaded_state_dict.items()
                if k in model_state_dict and model_state_dict[k].shape == v.shape
            }
            model_state_dict.update(pretrained_dict)
            model.load_state_dict(model_state_dict, strict=True)
        return model, criterion, postprocessors

    def run_ocr(self, model, data_loader_test):
        if os.path.exists(self.results_path):
            print(f"Skipping OCR evaluation because {self.results_path} already exists")
            with open(self.results_path, 'rb') as f:
                ocr_results = pickle.load(f)
            return ocr_results
        model.eval()
        batch_num = 0
        total_batches = len(data_loader_test)
        print("Total batches: ", total_batches)
        ocr_results = []
        for samples, targets in tqdm(data_loader_test, total=total_batches, desc="Processing", unit="batch"):
            batch_num += 1
            samples = samples.to(self.device)
            for t in targets:
                for k, v in t.items():
                    if not k == 'img_path':
                        t[k] = v.to(self.device)
    
            outputs = model(samples)
            for i in range(len(targets)):

                ocr_results.append({
                    'image_path': targets[i]['img_path'],
                    'orig_size': [int(targets[i]['orig_size'][1]), int(targets[i]['orig_size'][0])],
                    'pred_logits': outputs['pred_logits'][i].detach().cpu(),
                    'pred_boxes': outputs['pred_boxes'][i].detach().cpu()
                })
        with open(self.results_path, 'wb') as f:
            pickle.dump(ocr_results, f)

        return ocr_results
    
    def visualize(self, result):
        img_filepath = result["image_path"]
        pred_logits = result["pred_logits"]
        pred_bboxes = result["pred_boxes"]
        img_filename = img_filepath.split("/")[-1]
        
        bboxes_out_filename = "table_transformer_" + img_filename.replace(".jpg", "_bboxes.jpg")
        bboxes_out_filepath = os.path.join(self.debug_save_dir, bboxes_out_filename)

        img = Image.open(img_filepath)
        img_size = img.size

        m = pred_logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())
        pred_scores = list(m.values.detach().cpu().numpy())
        pred_bboxes = pred_bboxes.detach().cpu()
        pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

        for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
            if ((self.data_type == 'structure' and not label > 5)
                or (self.data_type == 'detection' and not label > 1)
                and score > 0.5):
                img = draw_color_bboxes([bbox], img)

        img.save(bboxes_out_filepath)


    def visualize_cells(self, bboxes, img_path):
        bboxes_lst = flatten_tensor(bboxes)
        img = Image.open(img_path)
        img = draw_color_bboxes(bboxes_lst, img)
        img_id = img_path.split("/")[-1].replace(".jpg", "_bboxes.jpg")
        bboxes_out_filename = "table_transformer_" + img_id
        bboxes_out_filepath = os.path.join(self.debug_save_dir, bboxes_out_filename)
        img.save(bboxes_out_filepath)

    def results_transform(self, ocr_results):
        transformed_results = []
        for result in ocr_results:
            if self.debug:
                self.visualize(result)
            trans_dir = {}
            img_path = result["image_path"]
            trans_dir['image_path'] = img_path
            img_filename = img_path.split("/")[-1]
            img_words_filepath = os.path.join(self.table_words_dir, img_filename.replace(".jpg", "_words.json"))
            #pdb.set_trace()
            trans_dir['img_words_path'] = img_words_filepath
            m = result["pred_logits"].softmax(-1).max(-1)
            pred_labels = list(m.indices.detach().cpu().numpy())
            pred_scores = list(m.values.detach().cpu().numpy())
            pred_bboxes = result["pred_boxes"].detach().cpu()
            pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, result['orig_size'])]
            with open(img_words_filepath, 'r') as f:
                true_page_tokens = json.load(f)
            _, pred_cells, _ = objects_to_cells(pred_bboxes, pred_labels, pred_scores,
                                true_page_tokens, structure_class_names,
                                structure_class_thresholds, structure_class_map)
            
            pred_relspan_grid = np.array(grits.cells_to_relspan_grid(pred_cells))
            pred_bbox_grid = np.array(grits.cells_to_grid(pred_cells, key='bbox'))
            pred_text_grid = np.array(grits.cells_to_grid(pred_cells, key='cell_text'), dtype=object)
            trans_dir['pred_cells'] = pred_cells
            trans_dir['pred_relspan_grid'] = pred_relspan_grid
            trans_dir['pred_bbox_grid'] = pred_bbox_grid
            trans_dir['pred_text_grid'] = pred_text_grid
            if self.debug:
                self.visualize_cells(pred_bbox_grid, img_path)
            transformed_results.append(trans_dir)
        return transformed_results


