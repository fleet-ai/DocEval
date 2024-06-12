# Document Understanding Evaluation Framework

This repository provides a structured framework for evaluating the performance of Document Understanding models on various tasks including Layout, Text Extraction (OCR), Table Detection, and Table Extraction. The current models supported are the Azure Layout model, Surya model, and the Table Transformer model from Microsoft Research. OCR on documents is fundamental to the functioning of many apps. Given the high costs associated with commercial models and the existence of accurate open-source models, benchmarking document understanding models on a variety of tasks and datasets is important in helping organizations optimize the performance of their systems while managing their resources effectively. Evaluating the performance of document understanding models can be challenging due to the lack of standardized output formats and the variety of tasks involved. This repository aims to address these challenges by providing a canonical structure for evaluating models across different tasks, datasets, and metrics.

<p float="left">
  <img src="https://github.com/fleet-ai/DocEval/blob/main/read_me_images/layout.png" width="400" />
  <img src="https://github.com/fleet-ai/DocEval/blob/main/read_me_images/table_extraction.png" width="400" /> 
</p>


## Contributing

This repo is a work in progress, so contributions to this evaluation framework are welcome! To add a model:

1. Examine how models for the same task are defined.
2. Implement the following methods in the model class:
   - `load_models`
   - `run_ocr`
   - `results_transform`
3. Check the docstring of other model classes for guidance on the output format of `results_transform`.
4. Import the model in the `models/__init__.py` file.
5. Alter the `prepare_models` method in the relevant task evaluation script in the `evaluations` folder to initialize and run the model.
6. Edit the `main.py` file to allow a new `model_name` to be inputted.

To add a new metric:

- Add the metric to the `run_eval` method in the evaluation class.

To add a new dataset:

- Convert the dataset to the canonical form in the evaluation class.

### Other Potential Models to Test

Consider testing the following models:

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (in progress)
- [LayoutLMv3](https://github.com/microsoft/unilm/blob/master/layoutlmv3) (I looked into this. I believe this requires bboxes and text, so I'm not sure how to fairly evaluate it.)
- [TrOCR](https://github.com/microsoft/unilm/tree/master/trocr) (requires line detection. The current text extraction evaluation combines line detection and OCR)
- [VGT](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT) (High reported performance on publaynet.)
- [VSR](https://github.com/hikopensource/DAVAR-Lab-OCR/tree/main/demo/text_layout/VSR) (High reported performance on publaynet.)
- [Textract](https://aws.amazon.com/textract/)(commercial)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OCR-D](https://ocr-d.de/)
- [Calamari OCR](https://github.com/Calamari-OCR/calamari)

### Datasets to Test

Consider testing the following datasets:

- [DocBank](https://doc-analysis.github.io/docbank-page/) (in progress)
- [SROIE](https://rrc.cvc.uab.es/?ch=13)
- [DocVQA](https://rrc.cvc.uab.es/?ch=17)
- [ICDAR 2019 cTDaR](https://cndplab-founder.github.io/cTDaR2019/)
- [TableBank](https://doc-analysis.github.io/tablebank-page/)

Looking for other text extraction datasets. There are a number of datasets for layout and tables, but fewer for text. (perhaps because its deemed to be easier?)

## Models

The framework currently supports the following models:

### Azure
The Azure Layout model is part of Azure Document Intelligence, and it performs all of the above tasks. It does not allow for a separation of tasks.
#### Tasks: Layout Analysis, Text Extraction, Table Detection, Table Extraction

### Surya
[Surya](https://github.com/VikParuchuri/surya) is a toolkit of open-source models (weights are conditionally licenced for commercial use depending on revenue) developed by Vik Paruchuri. It offers accurate text extraction and layout analysis capabilities.
#### Tasks: Layout Analysis, Text Extraction

### Table Transformer
[Table Transformer](https://github.com/microsoft/table-transformer) is an open-source model designed for table detection and extraction built by Microsoft Research. It is trained on the PubTables 1M dataset and the FinTabNet datasets.
#### Tasks: Table Detection, Table Extraction


## Usage

### Installation
1. Clone the repository:
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Download relevant datasets, and run relevant processing scripts. (only relevant for FinTabNet)
4. If using commercial models like Azure, create .env file in doceval folder with API keys. For example, for azure, the code searches for keys as follows
```python
endpoint = str(os.getenv('AZURE_API_URL'))
key = str(os.getenv('AZURE_API_KEY'))
```

4. Clone the Table-Transformer repo into the doceval folder:
```bash
https://github.com/microsoft/table-transformer.git
```

### Running Evaluations
To run evaluations, use the following command:
```bash
python -m doceval.main --evals <task> --model_names <models> --dataset_gt_name <dataset>
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
python -m doceval.main --evals layout --model_names Azure,Surya --dataset_gt_name publaynet --metrics precision,recall --max_doc 10 --visualize
```
```bash
python -m doceval.main --evals text_extraction --model_names Azure,Surya --dataset_gt_name vik_text_extraction_bench  --max_doc 10
```
```bash
python -m doceval.main --evals table_detection --model_names Azure,Table_Transformer --dataset_gt_name pubtables --max_doc 10 --visualize
```
```bash
python -m doceval.main --evals table_extraction --model_names Azure,Table_Transformer --dataset_gt_name fintabnet --max_doc 10 --visualize
```

NOTE: the evaluation will look in the data_ocr_results folder for OCR results. If a pkl file is found for the model+task, it will load in the file rather than rerun OCR. 

## Data
1. Clone the repo.
2. Download and process the relevant datasets:
  - Layout and text extraction datasets:
    - Run the `load_data` script in the `data` folder.
    - Specify the destination directories as:
      - `data/layout_bench/publaynet`
      - `data/text_extraction_bench/vik_text_extraction`
  - Table detection dataset:
    - Download the testing set of PubTables-1M dataset by following the directions [here](https://github.com/microsoft/table-transformer?tab=readme-ov-file).
    - Save the `PubTables-1M-Detection` folder in `data/table_detection_bench/pubtables`.
  - Table extraction dataset:
    - Download the testing set of the FinTabNet dataset by following the directions [here](https://developer.ibm.com/exchanges/data/all/fintabnet/#use-the-dataset4).
    - Save the dataset to `data/table_extraction_bench/fintabnet/fintabnet_raw`.
    - Run the `utils/process_fintabnet.py` file (from Microsoft).
    - Save the results to `data/table_extraction_bench/fintabnet/fintabnet_processed`.

## Table Transformer Model and Weights
1. If you want to use the Table Transformer model:
  -  Clone repo into the `doceval` folder:
      ```bash
      bash git clone https://github.com/microsoft/table-transformer.git
      ```
  - Download the weights for the detection model and the structure model.
  - Save the weights in the `model_weights/table-transformer` folder with the following names:
    - `detection_pubtables1m.pth`
    - `extraction_fintabnet.pth`

## Results

The evaluation results, including performance metrics and visualizations, will be stored in the `results` directory. The results are stored in a json file, and if the visualize flag is set, the resulting jpgs with bboxes will be saved in the benchmark directory. For table detection, follow the directions [here](https://github.com/microsoft/table-transformer?tab=readme-ov-file)


### Layout

<p float="left">
 <img src="https://github.com/fleet-ai/DocEval/blob/main/read_me_images/layout_eval.png" width="400" />
 <img src="https://github.com/fleet-ai/DocEval/blob/main/read_me_images/layout_eval_recall.png" width="400" /> 
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

<img src="https://github.com/fleet-ai/DocEval/blob/main/read_me_images/text_similarity.png" width="400" />

| Model | Text Similarity |
|-------|-----------------|
| Azure | 0.957           |
| Surya | 0.934           |

### Table Detection

<img src="https://github.com/fleet-ai/DocEval/blob/main/read_me_images/table_detection.png" width="400" />

| Model             | Average Precision (50%) | Average Recall (50%) |
|-------------------|-------------------------|----------------------|
| Table Transformer | 1.0                     | 1.0                  |
| Azure             | 0.992                   | 1.0                  |

### Table Extraction

<img src="https://github.com/fleet-ai/DocEval/blob/main/read_me_images/table_extraction_eval.png" width="400" />


## Additional Information
### Tasks

#### Layout Analysis
Layout analysis focuses on identifying and extracting the structural elements of a document, such as text blocks, figures, tables, and headings. Current metrics supported are precision and recall under a default of 50% coverage. Some challenges to canonicalization include differences in layout categories and how they can be accessed, differences in bbox units/format, and the amount of text contained within a text bbox. The only dataset currently supported is [PublayNet](https://github.com/ibm-aur-nlp/PubLayNet). Downloaded subset from [here](https://huggingface.co/datasets/vikp/publaynet_bench)

#### Text Extraction
Text extraction involves recognizing and extracting the textual content from an image or document. This task is fundamental to OCR and enables the conversion of visual information into machine-readable text. Some challenges to canonicalization include differences in the amount of text extracted (some models ignore headers, for example), and also how reading order can affect text similarity metrics. We use the Smith Waterman algorithm and fuzzy string matching to circumvent this. The only dataset currently supported was made by Vik Paruchuri [here](https://huggingface.co/vikp/text_recognizer_test).

#### Table Detection
Table detection aims to locate and identify the presence of tables within a document. Currently supported metrics are precision and recall under a default of 50% coverage. In addition to differences in bbox calculations, one issue to canonicalization stems from detecting tables split across consecutive pages. The only dataset currently supported is PubTables 1M. Follow directions [here](https://github.com/microsoft/table-transformer?tab=readme-ov-file) to download. I only downloaded the test set.

#### Table Extraction
Table extraction focuses on extracting the content and structure of identified tables. This task involves recognizing table cells, rows, columns, and their relationships. Specifically, we focus on three aspects of table extraction detailed in [GriTS](https://arxiv.org/abs/2203.12555): location, topology, and content. Cell topology recognition considers the layout of the cells, specifically the rows and columns each cell occupies over a two-dimensional grid. Cell content recognition considers the layout of cells and the text content of
each cell. Cell location recognition considers the layout of cells and the absolute coordinates of each cell within a document. This is the hardest output to standardize since not all models even extract table information details like the row and column span of a cell, models can make different decisions on what consistutes a row or column (both of which may be correct interpretations), and models differ in the size of the table cells they predict. Some models will default to predicting a grid while others will predict smaller bboxes around the cell content. Aligning cells in such a way that allows for a fair comparison of table contents is also challenging. The only dataset currently supported is FinTabNet from IBM. Follow directions [here](https://github.com/microsoft/table-transformer?tab=readme-ov-file) to download. I only downloaded the test set.


## To Do
- Make evaluations faster
- Improve table extraction
- Add models, datasets, and evaluation tasks

## Acknowledgements
This repo builds upon work in [Surya](https://github.com/VikParuchuri/surya), and [Microsoft Table Transformer](https://github.com/microsoft/table-transformer?tab=readme-ov-file)