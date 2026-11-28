# CASL-VAE

**Learning Structured Latent Variables from Unpaired Data for Semi-supervised Clustering and Paired Sample Generation**

CASL-VAE learns shared variation, target-specific subtypes, and within-subtype variation from **unpaired reference and target samples**. It supports subtype discovery, paired sample generation, and individual target-specific variation estimation. Training requires reference/target labels, but no subtype labels or matched pairs.

![CASL-VAE architecture and applications](docs/caslvaeoverview.png)

Common and salient encoders separate shared and target-specific factors. Holding the common latent fixed and replacing the salient code with the reference preset generates a paired reference estimate.

## Installation

```bash
conda env create -f environment.yml
conda activate caslvae
```

## Quick Start
Run from the repository root. Without arguments. Without arguments, the scripts below use the included [synthetic dataset](docs/synthetic_data) containing 2,000 unpaired reference-target subjects and 145 ROIs, with covariate correction disabled. 
```bash
# Compare category counts M = 2, 3, 4, 5, 6
python scripts/check_cluster_stability.py

# Train with one reference and two target categories
python scripts/run_CASLVAE.py
```
CUDA is used when available. 

## Input data
Provide one row per sample with domain label, covariates, and features:

```csv
subject_id,label,Age,Sex,ROI_1,ROI_2,ROI_3
sub-001,Background,63,1,3250,6100,4800
sub-002,Target,74,0,2900,5700,4300
```
Reference samples are denoted as **`Background`** label; all other labels are treated as target. Both groups are required. Age, sex, and intracranial volume can be used for optional covariate correction; categorical covariates must be encoded numerically.

Also provide a feature-list CSV. The feature_names column must contain the feature columns from the data file in the desired input order.
```csv
label,feature_names
1,ROI_1
2,ROI_2
3,ROI_3
```

## Training
To process your own data, supply the data and feature-list CSVs:

```bash
python scripts/check_cluster_stability.py --data-file data.csv --feature-names features.csv --m-list 2 3 4 5 --output stability_results

python scripts/run_CASLVAE.py --data-file data.csv --feature-names features.csv --ncategories 3 --output training_results
```
Stability results are saved in caslvae_cluster_stability_output/; the trained checkpoint `model_checkpoint.pt`, preprocessing parameters, and monitoring plots are saved in caslvae_training_output/.

For optional covariate correction and device selection: 
```bash
--covariate-correct --covariates Age Sex ICV --device cpu
```
**`ncategories` includes the reference category:** `3` means one reference and two target categories. Reference is Python index `0`; keep `nLatent_s_cat=1`.

An optional split column in data file specifies train, validation, or test assignments. Without it, the scripts create splits automatically. 

Use --help to see available arguments.

## Inference

The trained model can be used to infer category probabilities, common and salient embeddings, reconstructions, and paired reference estimates. Target-specific variation is the reconstruction minus its paired reference; the deviation index is its mean absolute value across features.

See [Example Script](docs/README.md) for model setup, checkpoint loading, and inference. Reuse training preprocessing and feature order for new samples. 

## Experiments
### Synthetic example and latent plots
The [145-ROI notebook](docs/CASLVAE_synthetic_145ROIs.ipynb) generates cross-sectional reference/target data and trains CASLVAE. See the notebook for step by step instruction.

### Non-neuroimaging datasets 
The [data preparation notebook](DataPreparation.ipynb) contains scripts for downloading and preprocessing the non-neuroimaging datasets used in the model generalization experiments.

### Neuroimaging datasets
Detailed instructions for neuroimaging data access and preprocessing are provided in the paper. The corresponding pipelines are not included in this repository due to data access restrictions and the complexity of the preprocessing procedures.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{
chintapalli2026caslvae,
title={{CASL}-{VAE}: Learning Structured Latent Variables from Unpaired Data for Semi-supervised Clustering and Paired Sample Generation},
author={Sai Spandana Chintapalli and Pratik Chaudhari and Christos Davatzikos},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=j3bc77a6Gn}
}
```

## License

Copyright (c) 2016 University of Pennsylvania. All rights reserved. See [https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html](https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html)

## Contact

Sai Spandana Chintapalli — [schinta@engineering.upenn.edu](mailto:schinta@engineering.upenn.edu)
