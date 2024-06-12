from doceval.models.base_model import OCR
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import load_model, load_processor
from surya.settings import settings
from pdf2image import convert_from_path
from PIL import Image
import pdb

import os
import pickle
base_dir = os.path.join(os.getcwd().split("DocEval")[0], "DocEval")

class SuryaLayout(OCR):
    def __init__(self, model_name: str, evals: list, layout_mapping: {}, checkpoint_paths = [settings.LAYOUT_MODEL_CHECKPOINT, settings.LAYOUT_MODEL_CHECKPOINT], results_path=os.path.join(base_dir, 'doceval/data/ocr_results/surya_layout.pkl'), write_output=True):
        super().__init__(model_name, evals, layout_mapping, checkpoint_paths, results_path, write_output)
    def load_models(self):
        model = load_model(checkpoint=self.checkpoint_paths[0])
        proc = load_processor(checkpoint=self.checkpoint_paths[1])
        det_model = load_model()
        det_processor = load_processor()
        return [det_model, det_processor, model, proc]
    
    def run_ocr(self, models, documents=[], document_paths=[]):
        """
        Args:
            models: tuple of models
            documents: list of PIL images
        
        """
        #NOTE Surya does not seem to support figures that span multiple pages
        if os.path.exists(self.results_path):
            print("Loading Surya layout results from file")
            return self.load_ocr_results()
        
        if not documents and not document_paths:
            raise ValueError("No documents provided")
        
        print("No Surya layout results found. Generating Surya layout results on documents...")
        det_model, det_processor, model, processor = models
        if document_paths and not documents:
            documents = [convert_from_path(path, dpi=72) for path in document_paths]
            
        surya_layout_results = []
        if len(documents) == 0:
            raise ValueError("No documents found")

        # Flatten the list of lists and keep track of page counts
        flat_documents = []
        page_counts = []
        for document in documents:
            flat_documents.extend(document)
            page_counts.append(len(document))

        # Process the flattened list of images
        flat_line_predictions = batch_text_detection(flat_documents, det_model, det_processor)
        flat_layout_predictions = batch_layout_detection(flat_documents, model, processor, flat_line_predictions)

        # Convert the flat layout predictions back to a list of lists
        start_index = 0
        for page_count in page_counts:
            end_index = start_index + page_count
            document_layout_predictions = flat_layout_predictions[start_index:end_index]
            surya_layout_results.append(document_layout_predictions)
            start_index = end_index
            
        if self.write_output:
            if self.results_path == '':
                raise ValueError("No results path provided to write the layout predictions to")
            with open(self.results_path, "wb") as f:
                pickle.dump(surya_layout_results, f)
        return surya_layout_results
        
    def results_transform(self, eval, results):
        """
        Args:
            eval: evaluation to run: "layout", "text"
            results: list of layout predictions
        """
        results_dict = {}
        if eval == "layout":
            for idx, surya_document_result in enumerate(results):
                num_pages = len(surya_document_result)
                results_dict[idx] = {}
                for label_name in self.layout_mapping.keys():
                    correct_labels = self.layout_mapping[label_name][self.model_name]
                    results_dict[idx][label_name] = {}
                    for page_num in range(1, num_pages+1):
                        results_dict[idx][label_name][page_num] = []
                        for bbox in surya_document_result[page_num-1].bboxes:
                            if bbox.label in correct_labels:
                                results_dict[idx][label_name][page_num].append({"coordinates": bbox.bbox, "span_pages": [page_num]})

            return results_dict
        if eval == "text":
            text_predictions = results