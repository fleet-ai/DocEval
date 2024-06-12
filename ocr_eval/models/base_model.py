import pickle
class OCR():
    def __init__(self, model_name: str, evals: list, layout_mapping={}, checkpoint_paths = '', results_path='', write_output = True):
        """
        Args:
            model_name: name of the model. must be the same as the model name in the layout_mapping
            evals: list of evaluations to run
            layout_mapping: mapping of layout labels to model labels
            checkpoint_paths: path to checkpoints
            results_path: path to save results
            write_output: whether to write output to disk
        """
        self.model_name = model_name
        self.evals = evals
        self.layout_mapping = layout_mapping
        self.results_path = results_path
        self.checkpoint_paths = checkpoint_paths
        self.write_output = write_output
        if "layout" in self.evals:
            if self.layout_mapping == {}:
                raise ValueError("Layout mapping is empty. Please provide a layout mapping.")

    def load_ocr_results(self):
        with open(self.results_path, "rb") as f:
            results = pickle.load(f)
        return results
    
    def load_models(self):
        raise NotImplementedError("Subclass must implement load_model method")
    
    def run_ocr(self, models, dataset):
        raise NotImplementedError("Subclass must implement get_ocr_results method")

    
    def results_transform(self, eval, results):
        raise NotImplementedError("Subclass must implement transform method")
    

        
