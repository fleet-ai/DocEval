from doceval.models.base_model import OCR

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.models import AnalyzeResult
from doceval.utils.utils import polygon_to_bbox, bbox_inch_to_dots, bbox_inch_to_pix, bbox_in_figure

import os
import pickle
import pdb
import string
base_dir = os.path.join(os.getcwd().split("DocEval")[0], "DocEval")
#NOTE Azure layout assumes figures and tabels named "Figures" and "Tables"
# Perhaps just extract figures and tables instead of breaking

class AzureTextExtraction(OCR):
    def __init__(self, model_name: str, evals: list, layout_mapping={}, checkpoint_paths = '', results_path=os.path.join(base_dir, 'doceval/data/ocr_results/azure_results.pkl'), write_output=True):
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
        print("No Azure layout results found. Generating Azure OCR results on documents...")
        azure_results = []
        for file_path in document_paths:
            with open(file_path, "rb") as f:
                poller = azure_model.begin_analyze_document(
                    "prebuilt-layout", analyze_request=f, content_type="application/octet-stream"
                )
            result: AnalyzeResult = poller.result()
            azure_results.append(result)
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
        results_dict = {}
        if eval == "text":
            for idx, azure_document_results in enumerate(results):
                results_dict[idx] = {}
                num_pages = len(azure_document_results.pages)
                for page in range(num_pages):
                    results_dict[idx][page+1] = {}
                    results_dict[idx][page+1]['bboxes'] = []
                    results_dict[idx][page+1]['text'] = []
                for paragraph in azure_document_results.paragraphs:
                    page = paragraph['boundingRegions'][0]['pageNumber']
                    par_text = paragraph.content.replace("\n", "")
                    for c in string.punctuation:
                        par_text  = par_text.replace(c, "")
                    par_words = par_text.split(" ")
                    while "" in par_words:
                        par_words.remove("")
                    #pdb.set_trace()
                    results_dict[idx][page]['text'] += par_words
                    polygon = []
                    for text_element in paragraph['boundingRegions']:
                        polygon += text_element.polygon
                    bbox = polygon_to_bbox(polygon)
                    if results_unit == "dots":
                        bbox = bbox_inch_to_dots(bbox)
                    elif results_unit == "pix":
                        bbox = bbox_inch_to_pix(bbox)
                    results_dict[idx][page]['bboxes'].append({"coordinates": bbox, "text": par_text})
            return results_dict
        elif eval == "layout":
            return results