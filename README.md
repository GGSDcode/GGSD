# GGSD

## Environment requirement
```
numpy
pandas
torch
torch-geometric
torch-sparse
tqdm
xlrd
pickle
logging
scipy
scikit-learn
networkx
```

## Dataset Download
You can find Cora, CiteSeer and PubMed dataset in the following url. Please download the graph dataset file that you are going to use.
https://github.com/tkipf/gcn/tree/master/gcn/data

## 1. Graph Dataset Preprocessing
First, create a directory for your graph dataset [DATASET_NAME] in ./data:
```
cd data
mkdir [DATASET_NAME]
```
Then, save the graph dataset as a .pt file in "raw_data_path" (described in the json file below).
After that, write a .json file ([DATASET_NAME].json) covering the following information:
```
{
    "name": "[DATASET_NAME]",
    "task_type": "multiclass",
    "header": "infer",
    "column_names": null,
    "file_type": "csv",
    "test_path": null,
    "raw_data_path": "[DATASET_PATH]/data.pt"
}
```
Put this .json file in the .Info directory.
Finally, run the following command to process your graph dataset:
```
python data/process_graph.py --dataname [DATASET_NAME]
```

## 2. VAE Training
```
python main.py --dataname [DATASET_NAME] --method VAE --mode train
```

## 3. GGSD Diffusion Training
```
python main.py --dataname [DATASET_NAME] --method GGSD --mode train
```

## 4. GGSD Diffusion Sampling
```
python main.py --dataname [DATASET_NAME] --method GGSD --mode sample --CL_guidance
```
