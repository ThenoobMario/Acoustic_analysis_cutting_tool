# Acoustic Analysis of Cutting Tool Wear

This repo contains the code used to analyse the Cutting Tool Wear Dataset. You can find the dataset on Kaggle titled, "[Cutting Tool Wear Audio Dataset](https://www.kaggle.com/datasets/nachiketsoni/cutting-tool-wear-audio-dataset)".

The dataset is a collection of **1488 audio samples** of **End Mill Cutters** of different levels of wear cutting through a piece of **mild steel** workpiece at two different RPMs using a **Vertical Milling Machine**. For more information about the dataset in particular, please refer to the [Kaggle link](https://www.kaggle.com/datasets/nachiketsoni/cutting-tool-wear-audio-dataset). Please note, there was no addition of background noise to the dataset.

## Steps to use the code

The code was tested with **Python3.9** as the interpreter.

- Clone this repo.
- Install dependencies using the below command:

```sh
pip install -r requirements.txt
```

- Download the dataset from Kaggle.
- Make changes to the `baseline.yaml` file related to the location of the dataset and neural network features.
- run `baseline.py` with the following command:
```sh
python baseline.py
```
- After completion, a `results.yaml` will be generated in the `results` directory and an `ROC Curve graph` will be generated inside the `model` directory.

> **NOTE:** I have included all the pickle files in the `paper_results` directory that can be used to reproduce the results mentioned in the paper.

## References and Citation 

The code is heavily inspired by the [MIMII_Baseline](https://github.com/MIMII-hitachi/mimii_baseline). 

If this work renders helpful to you, please consider citing our paper:

```bibtex
@INPROCEEDINGS{10461855,
  author={Soni, Nachiket and Kumar, Amit and Patel, Hardik},
  booktitle={2023 IEEE 11th Region 10 Humanitarian Technology Conference (R10-HTC)}, 
  title={Acoustic Analysis of Cutting Tool Vibrations of Machines for Anomaly Detection and Predictive Maintenance}, 
  year={2023},
  pages={43-46},
  doi={10.1109/R10-HTC57504.2023.10461855}}
```