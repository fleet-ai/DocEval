import os
import pickle
from PIL import Image
import requests
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from doceval.models.base_model import OCR
import string
#ONLY DOES LINE TEXT PREDICTION
base_dir = os.path.join(os.getcwd().split("DocEval")[0], "DocEval")

class TROCRTextExtraction(OCR):
    def __init__(self, model_name: str, evals: list, results_path=os.path.join(base_dir, 'doceval/data/ocr_results/trocr_text_extraction.pkl'), write_output=True):
        super().__init__(model_name=model_name, evals=evals, results_path=results_path, write_output=write_output)

    def load_models(self):
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
        return [processor, model]
    
    def run_ocr(self, models, documents=[], document_paths=[]):
        """
        Args:
            models: tuple of models
            documents: list of PIL images
        """
        if os.path.exists(self.results_path):
            print("Loading TrOCR text extraction results from file")
            return self.load_ocr_results()
        
        if not documents and not document_paths:
            raise ValueError("No documents provided")
        
        print("No TrOCR text extraction results found. Generating TrOCR text extraction results on documents...")
        processor, model = models

        if document_paths and not documents:
            documents = [convert_from_path(path)[0] for path in document_paths]
        pdb.set_trace()
        trocr_text_extraction_results = []
        for document in documents:
            pixel_values = processor(images=document, return_tensors="pt").pixel_values
            pdb.set_trace()
            generated_ids = model.generate(pixel_values)
            pdb.set_trace()
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            pdb.set_trace()
            trocr_text_extraction_results.append(generated_text)
        
        if self.write_output:
            if self.results_path == '':
                raise ValueError("No results path provided to write the layout predictions to")
            with open(self.results_path, "wb") as f:
                pickle.dump(trocr_text_extraction_results, f)
        
        return trocr_text_extraction_results
        
    def results_transform(self, eval, results):
        """
        Args:
            eval: evaluation to run: "text"
            results: ocr results from Tesseract
        Returns:
            results_dict: dictionary of results
            Text:
            {
                idx: {
                    page_num: {
                        "text": text
                    }
                }
            }
        """
        results_dict = {}
        if eval == "text":
            for idx, tesseract_document_result in enumerate(results):
                results_dict[idx] = {1: {}}  # Assuming only one page per document
                
                # Remove newline characters
                tesseract_document_result = tesseract_document_result.replace("\n", " ")
                
                # Remove punctuation characters
                for c in string.punctuation:
                    tesseract_document_result = tesseract_document_result.replace(c, "")
                
                # Split the text into words
                words = tesseract_document_result.split()
                
                # Remove empty strings from the list of words
                words = [word for word in words if word]
                
                results_dict[idx][1]['text'] = words
            
            return results_dict
        else:
            raise ValueError(f"Unsupported evaluation type: {eval}")

import os
import glob
import pdb
def extract_number(s):
    return int(''.join(filter(str.isdigit, s)))

if __name__ == "__main__":
    dataset_root = '/Users/judahengel/Documents/Fleet/evals/doceval/data/text_extraction_bench/vik_text_extraction_bench'
    documents = glob.glob(os.path.join(dataset_root, "*.pdf"))
    documents = sorted(documents, key=lambda x: extract_number(x))[:20]  # Limit to the first 20 documents

    model_name = "TrOCR"
    evals = ['text_extraction']  # Add the desired evaluation metrics here
    results_path = os.path.join(base_dir, 'doceval/data/ocr_results/trocr_text_extraction.pkl')
    
    trocr_text_extraction = TROCRTextExtraction(model_name=model_name, evals=evals, results_path=results_path)
    
    models = trocr_text_extraction.load_models()
    trocr_text_extraction_results = trocr_text_extraction.run_ocr(models, document_paths=documents)
    pdb.set_trace()
    print("TrOCR Text Extraction Results:")
    for i, result in enumerate(trocr_text_extraction_results):
        print(f"Document {i+1}:")
        print(result)
        print("---")

    