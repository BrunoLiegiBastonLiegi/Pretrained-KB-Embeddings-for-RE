# Pretrained Knowledge Base Embeddings for improved Sentential Relation Extraction

Implementation of what is discussed in the paper [url](url).

## Citation

If you find our work useful please consider citing us:

```
@something{Papaluca2022
...
}
```

## Repository Structure
### Core
- **model.py**: Implementation of the various models as *torch.nn.Module* objects.
- **trainer.py**: Trainer object that implements the training loop for the models in **model.py**.
- **dataset.py**: Implementation of the two objects used for preprocessing and preparing the data to be fed to the models.
- **run_experiment.py**: Main script to train and/or evaluate a model on one of the three datasets.
- **ner_schemes.py**: Simple implementation of the schemes for Named Entity Recognition (at the moment just BIOES is implemented).
### Results Visualization
- **plot_results.py**: Script for generating the plots shown in the paper.
- **utils.py**: Various helper functions mainly used by **plot_results.py**.
- **kg.py**: Little implementation of a Knowledge Graph object to visualize the graph of entities and relations.
### Datasets
Directories containing all the files relevant to the corpora, such as the train and test files, saved results and/or saved models

- **Wikidata/**: The *Wikidata* corpus, please refer to https://github.com/ansonb/RECON for the train and test files.
- **NYT/**: The *New York Times* corpus by [Riedel](https://www.researchgate.net/publication/220698997_Modeling_Relations_and_Their_Mentions_without_Labeled_Text), please refer to https://github.com/ansonb/RECON for the train and test files we used.
- **CONLL04/**: The *CONLL04* corpus, please refer https://github.com/lavis-nlp/spert for the train and test files we used ([shortcut](http://lavis.cs.hs-rm.de/storage/spert/public/datasets/conll04/)).

## Requirements

## Basic Usage

### Preparing the Data
For each corpus you can find the **preprocess.py** script to prepare the data in the format needed by **run_experiment.py** inside the relative directory (**Wikidata/**, **NYT/** or **CONLL04/**). In addition to the train/test file you are going to need the pretrained graph embedding file that you can find [here](https://torchbiggraph.readthedocs.io/en/latest/pretrained_embeddings.html), make sure to adapt it to have one or more (we split that in 4 due to RAM limitations) final **pickle**-serialized python dictionaries with keys and values given respectively by the Ids and the relative embeddings of the entities.
Once everything is set up, just run:
```
python preprocess.py PATH/_TO_THE/_FILE_TO_PROCESS.json PATH/_TO_THE/_PRETRAINED_EMBEDDING_FILE.pkl
```
by specifying which file to process (*PATH/_TO_THE/_FILE_TO_PROCESS.json*) and where the graph embedding file/s is/are located (*PATH/_TO_THE/_PRETRAINED_EMBEDDING_FILE.pkl*) .
You will obtain as output a **train.pkl**/**test.pkl** file ready to be fed to the **run_experiment.py** script.

### Training a Model

To train a model just run the **run_experiment.py** script:
```python
python run_experiment.py PATH/_TO_THE/_TRAIN_FILE.pkl PATH/_TO_THE/_TEST_FILE.pkl  
```
by specifying the location of the train and test files generated as explained above. 
You can specify various training options by editing the configuration file **conf.json**
```
{
    "n_epochs":	number of epochs to train (int),
    "n_exp": number of training runs (int),
    "batchsize": dimension of each batch (int),
    "lr": learning rate (float),
    "device": id of the gpu to use (int),
    "pretrained_lang_model": name of the pretrained language model (str),
    "evaluate": evaluate the performance? (bool),
    "negative_rel_class": label of the NA relation class (str)
}
```
 and running:
```
python run_experiment.py train.pkl test.pkl --conf PATH/_TO_/conf.json
```
If evaluate is set to true, you can also specify the json file where to save the results with:
```
python run_experiment.py train.pkl test.pkl --conf PATH/_TO_/conf.json --res_file results_file_name.json
```
this will store some information about the training settings and several performance metrics for each run, such as:
- Precision, Recall and F1 for each relation and on average
- The confusion matrix
- The micro *Precision-Recall* curve
