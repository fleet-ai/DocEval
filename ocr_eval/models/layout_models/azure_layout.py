from ocr_eval.models.base_model import OCR

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.models import AnalyzeResult
from ocr_eval.utils.utils import polygon_to_bbox, bbox_inch_to_dots, bbox_inch_to_pix, bbox_in_figure

import os
import pickle
import pdb
import time
base_dir = os.path.join(os.getcwd().split("evals")[0], "evals/ocr_eval")

class AzureLayout(OCR):
    def __init__(self, model_name: str, evals: list, layout_mapping={}, checkpoint_paths = '', results_path=os.path.join(base_dir, 'data/ocr_results/azure_results.pkl'), write_output=True):
        super().__init__(model_name, evals, layout_mapping, checkpoint_paths, results_path, write_output)

    def load_models(self, endpoint, key):
        azure_model = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        return [azure_model]
    
    def run_ocr(self, models, document_paths: list):
        """
        Args:
            models: list of models. Azure layout requires only one model
            document_paths: list of document paths to run the model on. Azure layout requires a binary stream as input.
            I've only implemented this for file paths, but I'll add support for a list of PIL Figures in the future.
        Returns:
            list of Azure layout results
        """
        azure_model = models[0]
        if os.path.exists(self.results_path):
            print("Loading Azure layout results from file")
            return self.load_ocr_results()
        print("No Azure layout results found. Generating Azure layout results on documents...")
        azure_results = []
        total_time = 0
        for file_path in document_paths:
            start_time = time.time()
            with open(file_path, "rb") as f:
                poller = azure_model.begin_analyze_document(
                    "prebuilt-layout", analyze_request=f, content_type="application/octet-stream"
                )
            result: AnalyzeResult = poller.result()
            ocr_time = time.time() - start_time
            total_time += ocr_time
            azure_results.append(result)
        print("Azure seconds per page {}".format(total_time/len(document_paths)))
        if self.write_output:
            with open(self.results_path, "wb") as f:
                pickle.dump(azure_results, f)
        return azure_results
    
    def results_transform(self, eval, results, results_unit = "dots"):
        """
        Args:
            eval: evaluation to run: "layout", "text"
            results: list of layout predictions
        """
        #pdb.set_trace()
        results_dict = {}
        if eval == "layout":
            for idx, azure_document_results in enumerate(results):
                num_pages = len(azure_document_results.pages)
                results_dict[idx] = {}
                for label_name in self.layout_mapping.keys():
                    results_dict[idx][label_name] = {}
                    if label_name in ["Figures", "Tables"]:
                        results_dict = self.get_figures_tables_bbox(results_dict, azure_document_results, label_name, idx, results_unit)
                        #pdb.set_trace()
                    elif label_name in ["Text", "Titles"]:
                        if "Figures" not in results_dict[idx].keys() or "Tables" not in results_dict[idx].keys():
                            raise ValueError("Figures and Tables must be extracted before Text or Titles. Assumes layout_mapping keys set to 'Figures' and 'Tables'")
                        results_dict = self.get_text_title_bbox(results_dict, azure_document_results, label_name, idx, results_unit)
                    else:
                        raise ValueError(f"Label {label_name} not supported")
            return results_dict
        elif eval == "text":
            return results
        
    def get_figures_tables_bbox(self, results_dict, azure_document_results, label, idx, results_unit = "dots"):
        """
        Args:
            results_dict: dict of layout predictions
            azure_document_results: Azure layout results
            label: label to get the bounding regions for
            idx: index of the document
            results_unit: unit of the results
        Logic for extracting figures and tables from Azure layout is identical.
        Filter out bounding regions that are in figures or tables.
        """
        fig_tab = []
        correct_labels = self.layout_mapping[label][self.model_name]
        num_pages = len(azure_document_results.pages)
        fig_tabs = []
        for fig_label in correct_labels:
            fig_tabs += azure_document_results.get(fig_label, [])
        for page in range(1, num_pages+1):
            results_dict[idx][label][page] = []
        #pdb.set_trace()
        for fig_tab in fig_tabs:
            # Combine figure bounding regions on the same page for the same figure. 
            # Keep separate bounding regions for a figure spanning more than one page.
            page_dict = {}
            for i in range(len(fig_tab['boundingRegions'])):
                if fig_tab['boundingRegions'][i]['pageNumber'] not in page_dict:
                    page_dict[fig_tab['boundingRegions'][i]['pageNumber']] = fig_tab['boundingRegions'][i]['polygon']
                else:
                    page_dict[fig_tab['boundingRegions'][i]['pageNumber']] += fig_tab['boundingRegions'][i]['polygon']
            #pdb.set_trace()
            for page, fig_polygon in page_dict.items():
                if page not in results_dict[idx][label].keys():
                    results_dict[idx][label][page] = []
                fig_bbox = polygon_to_bbox(fig_polygon)
                if results_unit == "dots":
                    bbox = bbox_inch_to_dots(fig_bbox)
                elif results_unit == "pix":
                    bbox = bbox_inch_to_pix(fig_bbox)
                #pdb.set_trace()
                results_dict[idx][label][page].append({"coordinates": bbox, "span_pages": list(page_dict.keys())})
        return results_dict
    
    def get_text_title_bbox(self, results_dict, azure_document_results, label, idx, results_unit = "dots"):
        """
        Args:
            results_dict: dict of layout predictions
            azure_document_results: Azure layout results
            label: label to get the bounding regions for
            idx: index of the document
            results_unit: unit of the results
        
        Text and titles both belong to the paragraphs field of Azure layout results, and must be extracted separately through role.
        Must check if the text is in a figure or table.
        """

        correct_labels = self.layout_mapping[label][self.model_name]
        num_pages = len(azure_document_results.pages)
        for page in range(1, num_pages+1):
            results_dict[idx][label][page] = []
        #pdb.set_trace()
        for paragraph in azure_document_results.get('paragraphs', []):
            #pdb.set_trace()
            if paragraph.get('role', None) in correct_labels:
                page_dict = {}
                #pdb.set_trace()
                for i in range(len(paragraph['boundingRegions'])):
                    if paragraph['boundingRegions'][i]['pageNumber'] not in page_dict:
                        page_dict[paragraph['boundingRegions'][i]['pageNumber']] = paragraph['boundingRegions'][i]['polygon']
                    else:
                        page_dict[paragraph['boundingRegions'][i]['pageNumber']] += paragraph['boundingRegions'][i]['polygon']
                #pdb.set_trace()
                for page, paragraph_polygon in page_dict.items():
                    #pdb.set_trace()
                    if page not in results_dict[idx][label].keys():
                        results_dict[idx][label][page] = []
                    paragraph_bbox = polygon_to_bbox(paragraph_polygon)
                    #pdb.set_trace()
                    if results_unit == "dots":
                        bbox = bbox_inch_to_dots(paragraph_bbox)
                    elif results_unit == "pix":
                        bbox = bbox_inch_to_pix(paragraph_bbox)
                    processed_tables = [tab_bbox.get('coordinates') for tab_bbox in results_dict[idx]['Tables'][page] if 'coordinates' in tab_bbox]
                    processed_figures = [fig_bbox.get('coordinates') for fig_bbox in results_dict[idx]['Figures'][page] if 'coordinates' in fig_bbox]
                    if not any([bbox_in_figure(bbox, tab) for tab in processed_tables] + [bbox_in_figure(bbox, fig) for fig in processed_figures]):
                        results_dict[idx][label][page].append({"coordinates": bbox, "span_pages": list(page_dict.keys())})
                    #results_dict[idx][label][page].append({"coordinates": bbox, "span_pages": list(page_dict.keys())})
        return results_dict



        
"""
layout_results_dict = {
    "document_idx": {
        "Text": {page_num: [{"coordinates": [x1, y1, x2, y2],
                                        "span_pages": [int]}}],
        "Figures": {page_num: [{"coordinates": [x1, y1, x2, y2],
                                        "span_pages": [int]}}],
        "Tables": {page_num: [{"coordinates": [x1, y1, x2, y2],
                                        "span_pages": [int]}}],
        "Titles": {page_num: [{"coordinates": [x1, y1, x2, y2],
                                        "span_pages": [int]}}]
    }
}

layout_mapping= {
    "Image": {"Publaynet": ["Figure"], "Surya": ["Picture", "Figure"], "Azure": ['figures']},
    "Table": {"Publaynet": ["Table"], "Surya": ["Table"], "Azure": ['tables']},
    "Text": {"Publaynet": ["Text", "List"], "Surya": ["Text", "Formula", "Footnote", "Caption", "List-item"], "Azure": [None, "formulaBlock", "footnote"]},
    "Title": {"Publaynet": ["Title"], "Surya": ["Section-header", "Title"], "Azure": ["title", "section_heading"]}}

"""

