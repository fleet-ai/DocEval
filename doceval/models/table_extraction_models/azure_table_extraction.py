import torch
import numpy as np
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

import json
import pdb

base_dir = os.path.join(os.getcwd().split("DocEval")[0], "DocEval")

def cell_bbox_doc_to_table(bbox, table_bbox, img_size):

    x_scale = img_size[0]/(table_bbox[2] - table_bbox[0])
    y_scale = img_size[1]/(table_bbox[3] - table_bbox[1])
    table_coords = [bbox[0] - table_bbox[0], bbox[1] - table_bbox[1], bbox[2] - table_bbox[0], bbox[3] - table_bbox[1]]
    
    scaled_bbox = [table_coords[0] * x_scale, table_coords[1] * y_scale, table_coords[2] * x_scale, table_coords[3] * y_scale]

    table_clipped = [max(table_coords[0], 0), max(table_coords[1], 0), min(table_coords[2], img_size[0]-1), min(table_coords[3], img_size[1]-1)]

    return table_clipped



def get_grid_bboxes(table, azure_table_bbox, table_bbox, img_size):
    rowCount = table['rowCount']
    columnCount = table['columnCount']
    bbox_grid = np.zeros((rowCount, columnCount, 4), dtype=int)

    for cell in table['cells']:
        rowIndex = cell['rowIndex']
        columnIndex = cell['columnIndex']
        rowSpan = cell.get('rowSpan', 1)
        columnSpan = cell.get('columnSpan', 1)

        polygon = []
        for region in cell['boundingRegions']:
            polygon.extend(region['polygon'])

        bbox_raw = polygon_to_bbox(polygon)

        jpg_scale_x = img_size[0] / (table_bbox[2]-table_bbox[0])
        jpg_scale_y = img_size[1] / (table_bbox[3]-table_bbox[1])

        pdf_scaled = bbox_inch_to_pix(bbox_raw)

        cropped_coords = [max(0, pdf_scaled[0] - table_bbox[0]), max(0, pdf_scaled[1] - table_bbox[1]), min(table_bbox[2], pdf_scaled[2] - table_bbox[0]), min(table_bbox[3], pdf_scaled[3] - table_bbox[1])]

        bbox = [cropped_coords[0] * jpg_scale_x, cropped_coords[1] * jpg_scale_y, cropped_coords[2] * jpg_scale_x, cropped_coords[3] * jpg_scale_y]

        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            continue

        #bbox = cell_bbox_doc_to_table(bbox, table_bbox, img_size)

        for i in range(rowSpan):
            for j in range(columnSpan):
                if rowIndex + i < rowCount and columnIndex + j < columnCount:
                    bbox_grid[rowIndex + i][columnIndex + j] = bbox

    return bbox_grid

def get_text_grid(table):
    rowCount = table['rowCount']
    columnCount = table['columnCount']
    text_grid = np.empty((rowCount, columnCount), dtype=object)

    for cell in table['cells']:
        rowIndex = cell['rowIndex']
        columnIndex = cell['columnIndex']
        rowSpan = cell.get('rowSpan', 1)
        columnSpan = cell.get('columnSpan', 1)

        content = cell.get('content', '')

        for i in range(rowSpan):
            for j in range(columnSpan):
                if rowIndex + i < rowCount and columnIndex + j < columnCount:
                    text_grid[rowIndex + i][columnIndex + j] = content

    return text_grid
def get_relspan_grid(table):
    """
    Convert an Azure table object to the matrix of grid cell features
    used for computing GriTS_Top.
    """
    num_rows = table['rowCount']
    num_columns = table['columnCount']
    cell_grid = np.zeros((num_rows, num_columns, 4))

    for cell in table['cells']:
        row_index = cell['rowIndex']
        column_index = cell['columnIndex']
        row_span = cell.get('rowSpan', 1)
        column_span = cell.get('columnSpan', 1)

        min_row_num = row_index
        min_column_num = column_index
        max_row_num = row_index + row_span
        max_column_num = column_index + column_span

        for row_num in range(min_row_num, max_row_num):
            for column_num in range(min_column_num, max_column_num):
                cell_grid[row_num][column_num] = np.array([
                    min_column_num - column_num,
                    min_row_num - row_num,
                    max_column_num - column_num,
                    max_row_num - row_num
                ])

    return cell_grid

def flatten_tensor(tensor):
    flattened = tensor.reshape(-1, tensor.shape[-1])
    return flattened.tolist()

def bbox_percent_coverage(bbox1, bbox2):
    # Extract coordinates from bounding boxes
    x1, y1, x2, y2 = bbox1
    a1, b1, a2, b2 = bbox2

    # Calculate the coordinates of the intersection rectangle
    left = max(x1, a1)
    top = max(y1, b1)
    right = min(x2, a2)
    bottom = min(y2, b2)

    # Check if the bounding boxes intersect
    if left < right and top < bottom:
        # Calculate the area of the intersection rectangle
        intersection_area = (right - left) * (bottom - top)

        # Calculate the area of the first bounding box
        bbox1_area = (x2 - x1) * (y2 - y1)

        # Calculate the percentage coverage
        percent_coverage = (intersection_area / bbox1_area) * 100
    else:
        # No intersection, so the percentage coverage is 0
        percent_coverage = 0

    return percent_coverage


class AzureTableExtraction:
    def __init__(self, dataset_root, table_words_dir, data_type, config_file, test_max_size, batch_size, num_workers, eval_pool_size, debug, results_path, debug_save_dir=os.path.join(base_dir, 'doceval/results/benchmark/table_extraction'), device="cpu"):
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
        self.from_pdf = True

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
                if "fintabnet" in self.dataset_root:
                    file_id = file_path.split('/')[-1].replace(".jpg", "")
                    company, year, page, num, table, table_num = file_id.split("_")
                    pdf_path = os.path.join(file_path.split("fintabnet_processed/")[0], "fintabnet_raw/pdf", company, year, f"page_{num}.pdf")
                    with open(pdf_path, "rb") as f:
                        poller = azure_model.begin_analyze_document(
                            "prebuilt-layout", analyze_request=f, content_type="application/octet-stream"
                        )
                    result: AnalyzeResult = poller.result()
                    result_dict['image_path'] = file_path
                    if len(result.tables) > int(table_num):
                        annotations_path = os.path.join(file_path.split("fintabnet_processed/")[0], "fintabnet_processed/FinTabNet.c_PDF_Annotations_JSON", f"{company}_{year}_page_{num}_tables.json")
                        annotation_dict = json.load(open(annotations_path))
                        table_bbox = annotation_dict[0]['pdf_table_bbox']
                        result_dict['table_bbox'] = table_bbox
                        coverages = []
                        for i, table in enumerate(result.tables):
                            azure_table_bbox = polygon_to_bbox(table['boundingRegions'][0]['polygon'])
                            azure_table_bbox = bbox_inch_to_pix(azure_table_bbox)
                            coverages.append(bbox_percent_coverage(table_bbox, azure_table_bbox))
                        if max(coverages) > 50:
                            result_dict['result'] = [result.tables[coverages.index(max(coverages))]]
                        else:
                            result_dict['result'] = []
                    else:
                        result_dict['result'] = []
                    ocr_results.append(result_dict)
                else:
                    with open(file_path, "rb") as f:
                        poller = azure_model.begin_analyze_document(
                            "prebuilt-layout", analyze_request=f, content_type="application/octet-stream"
                        )
                    result: AnalyzeResult = poller.result()
                    result_dict['image_path'] = file_path
                    if len(result.tables) > 0:
                        result_dict['result'] = [result.tables[0]]
                    else:
                        result_dict['result'] = []
                    ocr_results.append(result_dict)

        with open(self.results_path, 'wb') as f:
            pickle.dump(ocr_results, f)

        return ocr_results
    def visualize(self, result):
        img = Image.open(result['image_path'])
        img = draw_color_bboxes(result['bboxes'], img)
        img_id = "azure_" + result['image_path'].split("/")[-1].replace(".jpg", "_bboxes.jpg")
        img.save(os.path.join(self.debug_save_dir, img_id))
        
    def visualize_cells(self, bboxes, img_path):
        bboxes_lst = flatten_tensor(bboxes)
        img = Image.open(img_path)
        img = draw_color_bboxes(bboxes_lst, img)
        img_id = img_path.split("/")[-1].replace(".jpg", "_bboxes.jpg")
        bboxes_out_filename = "azure_" + img_id
        bboxes_out_filepath = os.path.join(self.debug_save_dir, bboxes_out_filename)
        img.save(bboxes_out_filepath)

    def results_transform(self, ocr_results):
        transformed_results = []
        for result in ocr_results:
            trans_dir = {}
            img_path = result["image_path"]
            trans_dir['image_path'] = img_path
            img_filename = img_path.split("/")[-1]
            img_words_filepath = os.path.join(self.table_words_dir, img_filename.replace(".jpg", "_words.json"))
            trans_dir['img_words_path'] = img_words_filepath
            trans_dir['pred_bbox_grid'] = [] 
            trans_dir['pred_text_grid'] = []
            trans_dir['pred_relspan_grid'] = []
            file_id = img_path.split('/')[-1].replace(".jpg", "")
            #company, year, page, num, table, table_num = file_id.split("_")
            #annotations_path = os.path.join(img_path.split("fintabnet_processed/")[0], "fintabnet_processed/FinTabNet.c_PDF_Annotations_JSON", f"{company}_{year}_page_{num}_tables.json")
            #annotation_dict = json.load(open(annotations_path))
            #table_bbox = annotation_dict[0]['pdf_table_bbox']
            table_bbox = result['table_bbox']
            img = Image.open(img_path)
            img_size = img.size
            print(file_id)
            if len(result['result'])==0:
                print("No result for {}".format(file_id))
            for table in result['result']:
                azure_table_bbox = polygon_to_bbox(table['boundingRegions'][0]['polygon'])
                trans_dir['pred_bbox_grid'] = get_grid_bboxes(table, azure_table_bbox, table_bbox, img_size)
                trans_dir['pred_text_grid'] = get_text_grid(table)
                trans_dir['pred_relspan_grid'] = get_relspan_grid(table)
            if self.debug:
                if len(trans_dir['pred_bbox_grid']) > 0:
                    try:
                        self.visualize_cells(trans_dir['pred_bbox_grid'], img_path)
                    except:
                        pdb.set_trace()
            transformed_results.append(trans_dir)
        return transformed_results