from .evaluations import LayoutEvaluation
from .evaluations import TextExtractionEvaluation
from .evaluations import TableDetectionEvaluation
from .evaluations import TableExtractionEvaluation
import os
import pickle
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--evals", type=str, help="Evaluation type: layout, text_extraction, table_detection, table_extraction")
parser.add_argument("--model_names", type=str, help="Model names: Azure, Surya, Table_Transformer")
parser.add_argument("--dataset_gt_name", type=str, help="Dataset ground truth name: Layout: publaynet, Text Detection: vik_text_extraction_bench, Table Detection: pubtables, Table Extraction: fintabnet")
parser.add_argument("--dataset_root_dir", type=str, default=None, help="OCR result paths")
parser.add_argument("--model_weights_dir", type=str, default = None, help="Directory containing model weights")
parser.add_argument("--metrics", type=str, default=None, help="Evaluation metrics: precision, recall, text similarity, grits_top, grits_loc, grits_con")
parser.add_argument("--max_doc", type=int, default=5, help="Maximum number of documents to evaluate")
parser.add_argument("--visualize", action="store_true", help="Visualize the evaluation results")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--eval_pool_size", type=int, default=4, help="Evaluation pool size")

SUPPORTED_MODELS = ["Azure", "Surya", "Table_Transformer"]
SUPPORTED_EVALS = ["layout", "text_extraction", "table_detection", "table_extraction"]
SUPPORTED_DATASETS = {"layout": ["publaynet"], "text_extraction": ["vik_text_extraction_bench"], "table_detection": ["pubtables"], "table_extraction": ["fintabnet"]}
SUPPORTED_METRICS = {"layout": ["precision", "recall"], "text_extraction": ["text similarity"], "table_detection": ["precision", "recall"], "table_extraction": ["grits_top", "grits_loc", "grits_con"]}

args = parser.parse_args()
evals = args.evals.split(",")
model_names = args.model_names.split(",")
dataset_gt_name = args.dataset_gt_name
dataset_root_dir = args.dataset_root_dir
model_weights_dir = args.model_weights_dir
if args.metrics == None:
    metrics = None
else:
    metrics = args.metrics.split(",")
max_doc = args.max_doc
visualize = args.visualize
num_workers = args.num_workers
batch_size = args.batch_size
eval_pool_size = args.eval_pool_size

LAYOUT_MAPPING_STANDARD= {
    "Figures": {"publaynet": ["Figure"], "Surya": ["Picture", "Figure"], "Azure": ['figures']},
    "Tables": {"publaynet": ["Table"], "Surya": ["Table"], "Azure": ['tables']},
    "Text": {"publaynet": ["Text", "List"], "Surya": ["Text", "Formula", "Footnote", "Caption", "List-item"], "Azure": [None, "formulaBlock", "footnote"]},
    "Titles": {"publaynet": ["Title"], "Surya": ["Section-header", "Title"], "Azure": ["title", 'sectionHeading']}}

base_dir = os.path.join(os.getcwd().split("OCR_Eval")[0], "OCR_Eval/ocr_eval")

def extract_number(s):
    return int(''.join(filter(str.isdigit, s)))

def run_layout():
    global dataset_root_dir
    global metrics
    ocr_result_paths = [os.path.join(base_dir, "data/ocr_results/" + model_name + "_layout_results.pkl") for model_name in model_names]
    if dataset_root_dir == None:
        if dataset_gt_name not in SUPPORTED_DATASETS["layout"]:
            raise ValueError(f"Dataset {dataset_gt_name} not supported by layout eval")
        else:
            dataset_root_dir = os.path.join(base_dir, "data/layout_bench", dataset_gt_name)
            print("No dataset root directory provided. Default dataset_root_dir: ", dataset_root_dir)
    layout_mapping = LAYOUT_MAPPING_STANDARD
    if metrics == None:
        metrics = ["precision", "recall"]
    else:
        for metric in metrics:
            if metric not in SUPPORTED_METRICS["layout"]:
                raise ValueError(f"Metric {metric} not supported")
    layout_eval = LayoutEvaluation(model_names, dataset_gt_name, dataset_root_dir, layout_mapping, max_doc, metrics)
    layout_eval.prepare_models(ocr_result_paths)
    layout_eval.run_eval()
    if visualize:
        layout_eval.visualize_eval()
    for model_name in model_names:
        results_ocr_path = os.path.join(base_dir, f"data/transformed_ocr_results/layout_{model_name}.json")
        with open(results_ocr_path, "w") as file:
            json.dump(layout_eval.models[model_name]['results'], file)
    results_path = os.path.join(base_dir, "results/layout_eval_results.json")
    with open(results_path, "w") as file:
        json.dump(layout_eval.eval_results, file)

def run_text_extraction():
    global dataset_root_dir
    global metrics
    if dataset_root_dir == None:
        if dataset_gt_name not in SUPPORTED_DATASETS["text_extraction"]:
            raise ValueError(f"Dataset {dataset_gt_name} not supported by text eval")
        else:
            dataset_root_dir = os.path.join(base_dir, "data/text_extraction_bench", dataset_gt_name)
    ocr_result_paths = [os.path.join(base_dir, f"data/ocr_results/text_extraction_{model_name}_results.pkl") for model_name in model_names]
    if metrics == None:
        metrics = ["text similarity"]
    else:
        for metric in metrics:
            if metric not in SUPPORTED_METRICS["text_extraction"]:
                raise ValueError(f"Metric {metric} not supported")
            
    text_extraction_eval = TextExtractionEvaluation(model_names, dataset_gt_name, dataset_root_dir, max_doc, metrics)
    text_extraction_eval.prepare_models(ocr_result_paths)
    text_extraction_eval.run_eval()

    results_path = os.path.join(base_dir, "results/text_extraction_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(text_extraction_eval.eval_results, f)
    for model_name in model_names:
        results_ocr_path = os.path.join(base_dir, f"data/transformed_ocr_results/text_extraction_{model_name}.json")
        with open(results_ocr_path, "w") as file:
            json.dump(text_extraction_eval.models[model_name]['results'], file)


def run_table_detection():
    global dataset_root_dir
    global metrics
    ocr_result_paths = [os.path.join(base_dir, f"data/ocr_results/table_detection_{model_name}_results.pkl") for model_name in model_names]
    if dataset_root_dir == None:
        if dataset_gt_name not in SUPPORTED_DATASETS["table_detection"]:
            raise ValueError(f"Dataset {dataset_gt_name} not supported by table detection eval")
        else:
            dataset_root_dir = os.path.join(base_dir, "data/table_detection_bench", dataset_gt_name)
    if metrics == None:
        metrics = ["precision", "recall"]
    else:
        for metric in metrics:
            if metric not in SUPPORTED_METRICS["table_detection"]:
                raise ValueError(f"Metric {metric} not supported")
    

    table_detection_eval = TableDetectionEvaluation(model_names, metrics, dataset_gt_name,
                                                    max_doc, batch_size, num_workers, eval_pool_size, ocr_result_paths, debug = visualize)
    
    print("Preparing models...")
    table_detection_eval.prepare_models()
    print("Running evaluation...")
    table_detection_eval.run_eval()

    results_path = os.path.join(base_dir, "results/table_detection_results.json")
    with open(results_path, "w") as f:
        json.dump(table_detection_eval.eval_results, f)
    print("Evaluation results saved to:", results_path)
    print(table_detection_eval.eval_results)

    for model_name in model_names:
        results_ocr_path = os.path.join(base_dir, f"data/transformed_ocr_results/table_detection_{model_name}.pkl")
        with open(results_ocr_path, "wb") as file:
            pickle.dump(table_detection_eval.models, file) 


def run_table_extraction():
    global dataset_root_dir
    global metrics
    ocr_result_paths = [os.path.join(base_dir, f"data/ocr_results/table_extraction_{model_name}_results.pkl") for model_name in model_names]
    if dataset_root_dir == None:
        if dataset_gt_name not in SUPPORTED_DATASETS["table_extraction"]:
            raise ValueError(f"Dataset {dataset_gt_name} not supported by table extraction eval")
        else:
            dataset_root_dir = os.path.join(base_dir, "data/table_extraction_bench", dataset_gt_name)
    if metrics == None:
        metrics = ["grits_top", "grits_loc", "grits_con"]
    else:
        for metric in metrics:
            if metric not in SUPPORTED_METRICS["table_extraction"]:
                raise ValueError(f"Metric {metric} not supported")
    table_eval = TableExtractionEvaluation(model_names, metrics, dataset_gt_name, max_doc, batch_size, num_workers, eval_pool_size, ocr_result_paths, debug = visualize)
    print("Preparing models...")
    table_eval.prepare_models()
    print("Running evaluation...")
    table_eval.run_eval()
    results_path = os.path.join(base_dir, "results/table_extraction_results.json")
    with open(results_path, "w") as f:
        json.dump(table_eval.eval_results, f)
    print("Evaluation results saved to:", results_path)
    print(table_eval.eval_results)

    for model_name in model_names:
        results_ocr_path = os.path.join(base_dir, f"data/transformed_ocr_results/table_extraction_{model_name}.pkl")
        with open(results_ocr_path, "wb") as file:
            pickle.dump(table_eval.models, file) 


if __name__ == "__main__":
    if "layout" in evals:
        run_layout()

    if "text_extraction" in evals:
        run_text_extraction()


    if "table_detection" in evals:
        run_table_detection()


    if "table_extraction" in evals:
        run_table_extraction()

