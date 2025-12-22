## Experimental Configuration Description

For the complete experimental configuration (including hyperparameters, hardware information, etc.), please refer to: 

- Base Configuration File: `configs/base_config.json` 
- Dataset-specific Configuration Files: `configs/dataset_specific/` 
- Detailed Description of Configuration Parameters: `APPENDIX.md`

## Quickstart

this project is fully tested under python 3.10, it is recommended that you set the Python version to 3.10.

1. Requirements

Given a python environment (**note**: this project is fully tested under python 3.10), install the dependencies with the following command:

```
pip install -r requirements.txt
```

​2. Data preparation

You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/1ycq7ufOD2eFOjDkjr0BfSg?pwd=bpry). Then place the downloaded data under the folder `./dataset`.

​3. Train and evaluate model

We provide the experiment scripts for D-TIME under the folder `./scripts/multivariate_forecast`. For example you can reproduce all the experiment results as the following script:

```
sh ./scripts/multivariate_forecast/ETTh1_script/DTIME.sh
```

 
