import torch
import argparse
from torch.utils.data import DataLoader
from doceval.datasets.table_dataset import PDFTablesDataset, get_structure_transform, get_detection_transform
from doceval.utils.utils import collate_fn
import os
import json
from tqdm import tqdm
import pdb
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from doceval.utils.utils import draw_color_bboxes, polygon_to_bbox, bbox_inch_to_dots, bbox_inch_to_pix, bbox_in_figure


from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.models import AnalyzeResult

from dotenv import load_dotenv
base_dir = os.path.join(os.getcwd().split("DocEval")[0], "DocEval")

class AzureTableDetection:
    def __init__(self, dataset_root, table_words_dir, data_type, config_file, test_max_size, batch_size, num_workers, eval_pool_size, debug, results_path, debug_save_dir=os.path.join(base_dir, 'doceval/results/benchmark/table_detection'), device="cpu"):
        self.dataset_root = dataset_root
        self.table_words_dir = table_words_dir
        self.data_type = data_type
        self.config_file = config_file
        self.test_max_size = test_max_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_pool_size = eval_pool_size
        self.device = device
        self.debug = debug
        self.results_path = results_path
        self.debug_save_dir = debug_save_dir
        self.args = self.create_args()
        print(self.dataset_root)

    def create_args(self):
        cmd_args = argparse.Namespace()
        cmd_args.data_root_dir = self.dataset_root
        cmd_args.config_file = self.config_file
        cmd_args.data_type = self.data_type
        cmd_args.table_words_dir = self.table_words_dir
        cmd_args.batch_size = self.batch_size
        cmd_args.num_workers = self.num_workers
        cmd_args.test_max_size = self.test_max_size
        cmd_args.eval_pool_size = self.eval_pool_size
        cmd_args.device = self.device
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
    
    def load_models(self, endpoint, key):
        azure_model = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        return [azure_model]

    def run_ocr(self, models, data_loader_test):
        if os.path.exists(self.results_path):
            print(f"Skipping OCR evaluation because {self.results_path} already exists")
            with open(self.results_path, 'rb') as f:
                ocr_results = pickle.load(f)
            return ocr_results

        azure_model = models[0]
        ocr_results = []

        for samples, targets in tqdm(data_loader_test, total=len(data_loader_test), desc="Processing", unit="batch"):
            for target in targets:
                result_dict = {}
                file_path = target['img_path']
                with open(file_path, "rb") as f:
                    poller = azure_model.begin_analyze_document(
                        "prebuilt-layout", analyze_request=f, content_type="application/octet-stream"
                    )
                result: AnalyzeResult = poller.result()
                result_dict['image_path'] = file_path
                result_dict['result'] = result
                ocr_results.append(result_dict)

        with open(self.results_path, 'wb') as f:
            pickle.dump(ocr_results, f)

        return ocr_results
    def visualize(self, result):
        img = Image.open(result['image_path'])
        img = draw_color_bboxes(result['bboxes'], img)
        img_id = "azure_" + result['image_path'].split("/")[-1].replace(".jpg", "_bboxes.jpg")
        img.save(os.path.join(self.debug_save_dir, img_id))

    def results_transform(self, ocr_results):
        transformed_results = []
        for result in ocr_results:
            trans_dir = {}
            trans_dir['image_path'] = result["image_path"]
            trans_dir['bboxes'] = []
            for table in result['result'].tables:
                polygon = []
                for region in table['boundingRegions']:
                    polygon.extend(region['polygon'])
                bbox = polygon_to_bbox(polygon)
                #no unit conversion because we are using jpg for azure ocr. Only outputs inches for pdf.
                #bbox = bbox_inch_to_dots(bbox)
                trans_dir['bboxes'].append(bbox)
            if self.debug:
                self.visualize(trans_dir)
            transformed_results.append(trans_dir)
        return transformed_results



