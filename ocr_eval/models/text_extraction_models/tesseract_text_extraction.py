import os
import pickle
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from ocr_eval.models.base_model import OCR
import glob
import pdb
# ONLY DOES LINE TEXT PREDICTION
base_dir = os.path.join(os.getcwd().split("evals")[0], "evals")

class TesseractTextExtraction(OCR):
    def __init__(self, model_name: str, evals: list, results_path=os.path.join(base_dir, 'ocr_eval/data/ocr_results/tesseract_text_extraction.pkl'), write_output=True):
        super().__init__(model_name=model_name, evals=evals, results_path=results_path, write_output=write_output)

    def load_models(self):
        # No need to load any models for Tesseract
        return []
    
    def run_ocr(self, models, documents=[], document_paths=[]):
        """
        Args:
            models: tuple of models (empty for Tesseract)
            documents: list of PIL images
        """
        if os.path.exists(self.results_path):
            print("Loading Tesseract text extraction results from file")
            return self.load_ocr_results()
        
        if not documents and not document_paths:
            raise ValueError("No documents provided")
        
        print("No Tesseract text extraction results found. Generating Tesseract text extraction results on documents...")
        
        if document_paths and not documents:
            documents = [convert_from_path(path)[0] for path in document_paths]
        
        tesseract_text_extraction_results = []
        for document in documents:
            # Perform OCR using Tesseract
            text = pytesseract.image_to_string(document)
            tesseract_text_extraction_results.append(text)
        
        if self.write_output:
            if self.results_path == '':
                raise ValueError("No results path provided to write the layout predictions to")
            with open(self.results_path, "wb") as f:
                pickle.dump(tesseract_text_extraction_results, f)
        
        return tesseract_text_extraction_results

def extract_number(s):
    return int(''.join(filter(str.isdigit, s)))

if __name__ == "__main__":
    dataset_root = '/Users/judahengel/Documents/Fleet/evals/ocr_eval/data/text_extraction_bench/vik_text_extraction_bench'
    documents = glob.glob(os.path.join(dataset_root, "*.pdf"))
    documents = sorted(documents, key=lambda x: extract_number(x))[:20]  # Limit to the first 20 documents

    model_name = "Tesseract"
    evals = ['text_extraction']  # Add the desired evaluation metrics here
    results_path = os.path.join(base_dir, 'ocr_eval/data/ocr_results/tesseract_text_extraction.pkl')
    
    trocr_text_extraction = TesseractTextExtraction(model_name=model_name, evals=evals, results_path=results_path)
    
    models = trocr_text_extraction.load_models()
    trocr_text_extraction_results = trocr_text_extraction.run_ocr(models, document_paths=documents)
    pdb.set_trace()