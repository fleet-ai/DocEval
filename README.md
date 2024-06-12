# OCR Evaluation Framework

This repository provides a structured framework for evaluating the performance of OCR (Optical Character Recognition) models on various tasks including Layout, Text Extraction, Table Detection, and Table Extraction. The current models supported are the Azure Layout model, Surya model, and the Table Transformer model from Microsoft Research. 

<p float="left">
  <img src="https://github.com/fleet-ai/OCR_Eval/blob/main/read_me_images/layout.png" width="400" />
  <img src="https://github.com/fleet-ai/OCR_Eval/blob/main/read_me_images/table_extraction.png" width="400" /> 
</p>

## Introduction

OCR on documents is fundamental to the functioning of many apps. Given the high costs associated with commercial OCR models and the existence of accurate open-source models, benchmarking OCR models on a variety of tasks and datasets is important in helping organizations optimize the performance of their systems while managing their resources effectively. Evaluating the performance of OCR models can be challenging due to the lack of standardized output formats and the variety of tasks involved. This repository aims to address these challenges by providing a canonical structure for evaluating OCR models across different tasks, datasets, and metrics.

## Tasks

### Layout Analysis
Layout analysis focuses on identifying and extracting the structural elements of a document, such as text blocks, figures, tables, and headings. Current metrics supported are precision and recall under a default of 50% coverage. Some challenges to canonicalization include differences in layout categories and how they can be accessed, differences in bbox units/format, and the amount of text contained within a text bbox. The only dataset currently supported is [PublayNet](https://github.com/ibm-aur-nlp/PubLayNet). Downloaded subset from [here](https://huggingface.co/datasets/vikp/publaynet_bench)

### Text Extraction
Text extraction involves recognizing and extracting the textual content from an image or document. This task is fundamental to OCR and enables the conversion of visual information into machine-readable text. Some challenges to canonicalization include differences in the amount of text extracted (some models ignore headers, for example), and also how reading order can affect text similarity metrics. We use the Smith Waterman algorithm and fuzzy string matching to circumvent this. The only dataset currently supported was made by Vik Paruchuri [here](https://huggingface.co/vikp/text_recognizer_test).

### Table Detection
Table detection aims to locate and identify the presence of tables within a document. Currently supported metrics are precision and recall under a default of 50% coverage. In addition to differences in bbox calculations, one issue to canonicalization stems from detecting tables split across consecutive pages. The only dataset currently supported is PubTables 1M. Follow directions [here](https://github.com/microsoft/table-transformer?tab=readme-ov-file) to download. I only downloaded the test set.

### Table Extraction
Table extraction focuses on extracting the content and structure of identified tables. This task involves recognizing table cells, rows, columns, and their relationships. Specifically, we focus on three aspects of table extraction detailed in [GriTS](https://arxiv.org/abs/2203.12555): location, topology, and content. Cell topology recognition considers the layout of the cells, specifically the rows and columns each cell occupies over a two-dimensional grid. Cell content recognition considers the layout of cells and the text content of
each cell. Cell location recognition considers the layout of cells and the absolute coordinates of each cell within a document. This is the hardest output to standardize since not all models even extract table information details like the row and column span of a cell, models can make different decisions on what consistutes a row or column (both of which may be correct interpretations), and models differ in the size of the table cells they predict. Some models will default to predicting a grid while others will predict smaller bboxes around the cell content. Aligning cells in such a way that allows for a fair comparison of table contents is also challenging. The only dataset currently supported is FinTabNet from IBM. Follow directions [here](https://github.com/microsoft/table-transformer?tab=readme-ov-file) to download. I only downloaded the test set.

## Models

The framework currently supports the following OCR models:

### Azure
The Azure Layout model is part of Azure Document Intelligence, and it performs all of the above tasks. It does not allow for a separation of tasks and costs $10 per 1000 pages.

### Surya
[Surya](https://github.com/VikParuchuri/surya) is an open-source OCR model (weights are conditionally licenced for commercial use depending on revenue) developed by Vik Paruchuri. It offers accurate text extraction and layout analysis capabilities.

### Table Transformer
[Table Transformer](https://github.com/microsoft/table-transformer) is an open-source model designed for table detection and extraction built by Microsoft Research. It is trained on the PubTables 1M dataset and the FinTabNet datasets.

## Usage

### Installation
1. Clone the repository:
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Download relevant datasets, and run relevant processing scripts. (only relevant for FinTabNet)
4. If using commercial models like Azure, create .env file in root directory with API keys. For example, for azure, the code searches for keys as follows
```python
endpoint = str(os.getenv('AZURE_API_URL'))
key = str(os.getenv('AZURE_API_KEY'))
```
### Running Evaluations
To run evaluations, use the following command:
```bash
python -m ocr_eval.main --evals <task> --model_names <models> --dataset_gt_name <dataset>
```
Replace `<task>`, `<models>`, and `<dataset>` with the desired values.

Additional flags:
- `--dataset_root_dir`: Specify the root directory for the dataset. Will search for default directory.
- `--model_weights_dir`: Specify the directory containing model weights.
- `--metrics`: Specify the evaluation metrics (e.g., precision, recall, text similarity).
- `--max_doc`: Specify the maximum number of documents to evaluate.
- `--visualize`: Enable visualization of evaluation results.
Example usages:
```bash
python -m ocr_eval.main --evals layout --model_names Azure,Surya --dataset_gt_name publaynet --metrics precision,recall --max_doc 10 --visualize
```
```bash
python -m ocr_eval.main --evals text_extraction --model_names Azure,Surya --dataset_gt_name vik_text_extraction_bench  --max_doc 10
```
```bash
python -m ocr_eval.main --evals table_detection --model_names Azure,Table_Transformer --dataset_gt_name pubtables --max_doc 10 --visualize
```
```bash
python -m ocr_eval.main --evals table_extraction --model_names Azure,Table_Transformer --dataset_gt_name fintabnet --max_doc 10 --visualize
```


## Data
After cloning the repo, the relevant datasets must be downloaded and processed. The the layout and text extraction datasets can be downloaded by running the load_data script in the data folder; specify the destination directories as data/layout_bench/publaynet and data/text_extraction_bench/vik_text_extraction. For table detection, download the testing set of PubTables-1M dataset by following the directions [here](https://github.com/microsoft/table-transformer?tab=readme-ov-file) and save the PubTables-1M-Detection folder in data/table_detection_bench/pubtables folder. For table extraction, download the testing set of the FinTabNet dataset by following the directions [here](https://developer.ibm.com/exchanges/data/all/fintabnet/#use-the-dataset4) and save to data/table_extraction_bench/fintabnet/fintabnet_raw. Then run the utils/process_fintabnet.py file (from Microsoft) and save the results to data/table_extraction_bench/fintabnet/fintabnet_processed.

## Results

The evaluation results, including performance metrics and visualizations, will be stored in the `results` directory. The results are stored in a json file, and if the visualize flag is set, the resulting jpgs with bboxes will be saved in the benchmark directory. For table detection, follow the directions [here](https://github.com/microsoft/table-transformer?tab=readme-ov-file)


### Layout

<p float="left">
 <img src="https://github.com/fleet-ai/OCR_Eval/blob/main/read_me_images/layout_eval.png" width="400" />
 <img src="https://github.com/fleet-ai/OCR_Eval/blob/main/read_me_images/layout_eval_recall.png" width="400" /> 
</p>

**Azure Average:**

| Metric    | Figures | Tables | Text  | Titles | Total |
|-----------|---------|--------|-------|--------|-------|
| Precision | 1.0     | 1.0    | 0.897 | 0.955  | 0.914 |
| Recall    | 1.0     | 1.0    | 0.941 | 0.929  | 0.941 |

**Surya Average:**

| Metric    | Figures | Tables | Text  | Titles | Total |
|-----------|---------|--------|-------|--------|-------|
| Precision | 0.956   | 1.0    | 0.917 | 0.941  | 0.925 |
| Recall    | 1.0     | 1.0    | 0.959 | 0.893  | 0.947 |

### Text Extraction

<img src="https://github.com/fleet-ai/OCR_Eval/blob/main/read_me_images/text_similarity.png" width="400" />

| Model | Text Similarity |
|-------|-----------------|
| Azure | 0.957           |
| Surya | 0.934           |

### Table Detection

<img src="https://github.com/fleet-ai/OCR_Eval/blob/main/read_me_images/table_detection.png" width="400" />

| Model             | Average Precision (50%) | Average Recall (50%) |
|-------------------|-------------------------|----------------------|
| Table Transformer | 1.0                     | 1.0                  |
| Azure             | 0.992                   | 1.0                  |

### Table Extraction

<img src="https://github.com/fleet-ai/OCR_Eval/blob/main/read_me_images/table_extraction_eval.png" width="400" />


## Contributing

Contributions to this OCR evaluation framework are welcome! To add a model, you should examine how models for the same task are defined. The class should implement methods: load_models, run_ocr, and results_transform. The other model classes should have a docstring that defines what the output of results_transform should look like that should help guide the canonicalization process. You should import the model in the models/__init__.py file in the models You should also alter the prepare_models method in the relevant task evaluation script in the evaluations folder to initialize and run the model. You should also edit the main.py file to allow a new model_name to be inputted. To add a new metric, you can add it to the run_eval method in the evaluation class. To add a new dataset, you need to convert it to the canonical form in the evaluation class.
