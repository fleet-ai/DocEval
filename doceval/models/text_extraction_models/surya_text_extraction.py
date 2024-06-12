from doceval.models.base_model import OCR
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import load_model as load_detection_model
from surya.model.detection.segformer import load_processor as load_detection_processor
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.ocr import run_recognition
from surya.settings import settings
from pdf2image import convert_from_path
from PIL import Image
import pdb
import string

import os
import pickle
base_dir = os.path.join(os.getcwd().split("DocEval")[0], "DocEval")

class SuryaTextExtraction(OCR):
    def __init__(self, model_name: str, evals: list, results_path=os.path.join(base_dir, 'doceval/data/ocr_results/surya_text_extraction.pkl'), write_output=True):
        super().__init__(model_name=model_name, evals=evals, results_path=results_path, write_output=write_output)

    def load_models(self):
        det_model = load_detection_model()
        det_processor = load_detection_processor()
        model = load_recognition_model()
        processor = load_recognition_processor()
        return [det_model, det_processor, model, processor]
    
    def run_ocr(self, models, documents=[], document_paths=[]):
        """
        Args:
            models: tuple of models
            documents: list of PIL images
        
        """
        if os.path.exists(self.results_path):
            print("Loading Surya text extraction results from file")
            return self.load_ocr_results()
        
        print("No Surya text extraction results found. Generating Surya text extraction results on documents...")
        det_model, det_processor, model, processor = models

        if not documents and not document_paths:
            raise ValueError("No documents provided")

        if document_paths and not documents:
            documents = [convert_from_path(path) for path in document_paths]

        surya_text_extraction_results = []
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
        flat_bboxes_surya = [[element.bbox for element in flat_line_predictions[i].bboxes] for i in range(len(flat_line_predictions))]
        flat_text_extraction = run_recognition(flat_documents, [['af']] * len(flat_documents), model, processor, bboxes=flat_bboxes_surya)

        # Convert the flat text extraction results back to a list of lists
        start_index = 0
        for page_count in page_counts:
            end_index = start_index + page_count
            document_text_extraction = flat_text_extraction[start_index:end_index]
            surya_text_extraction_results.append(document_text_extraction)
            start_index = end_index

        if self.write_output:
            if self.results_path == '':
                raise ValueError("No results path provided to write the layout predictions to")
            with open(self.results_path, "wb") as f:
                pickle.dump(surya_text_extraction_results, f)
        return surya_text_extraction_results
        
    def results_transform(self, eval, results):
        """
        Args:
            eval: evaluation to run: "layout", "text"
            results: ocr results from model
        Returns:
            results_dict: dictionary of results
            Text:
            {
                idx: {
                    page_num: {
                        "text": text,
                        "bboxes": [{"coordinates": bbox, "text": text}]
                    }
                }
            }
            
        """
        results_dict = {}
        if eval == "text":
            for idx, surya_document_result in enumerate(results):
                results_dict[idx] = {}
                for page_idx, page in enumerate(surya_document_result):
                    results_dict[idx][page_idx+1] = {}
                    results_dict[idx][page_idx+1]['bboxes'] = []
                    page_words = []
                    for line in page.text_lines:
                        line_text = line.text.replace("\n", "")
                        for c in string.punctuation:
                            line_text = line_text.replace(c, "")
                        line_words = line_text.split(" ")
                        while "" in line_words:
                            line_words.remove("")
                        page_words += line_words
                        results_dict[idx][page_idx+1]['bboxes'].append({"coordinates": line.bbox, "text": line_text})
                    results_dict[idx][page_idx+1]['text'] = page_words
            return results_dict
        if eval == "layout":
            text_predictions = results

