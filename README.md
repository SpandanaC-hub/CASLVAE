# CASL-VAE

**Contrastive Analysis with Structured Latents via Variational Autoencoder**

CASL-VAE learns shared variation, target-specific subtypes, and within-subtype variation from **unpaired reference and target samples**. It supports subtype discovery, paired sample generation, and individual target-specific variation estimation. Training requires reference/target labels, but no subtype labels or matched pairs.

![CASL-VAE architecture and applications](docs/caslvaeFig1_redesign.pdf)

Common and salient encoders separate shared and target-specific factors. Holding the common latent fixed and replacing the salient code with the reference preset generates a paired reference estimate.

## Installation

```bash
conda env create -f environment.yml
conda activate caslvae
```

## Input data

Provide one row per sample with numeric features and a group label:

```csv
subject_id,label,ROI_1,ROI_2,ROI_3
sub-001,Background,3250,6100,4800
sub-002,Target,2900,5700,4300
```

Reference samples must use **`Background`**; all other labels are treated as target. Both groups are required. Age, sex, and intracranial volume can be used for optional covariate correction; categorical covariates must be encoded numerically.

## Training

Edit the dataset paths, feature/covariate lists, and output directory in the scripts, then run from the repository root:

```bash
# Compare category counts using clustering stability
python scripts/check_cluster_stability_realdata.py

# Train with the selected opt['ncategories']
python scripts/run_CASLVAE_realdata.py
```

**`ncategories` includes the reference category:** `3` means one reference and two target categories. Reference is Python index `0`; keep `nLatent_s_cat=1`.

Training saves `model_checkpoint.pt`, preprocessing parameters, and monitoring plots. 

## Inference

The trained model can be used to infer category probabilities, common and salient embeddings, reconstructions, and paired reference estimates. Target-specific variation is the reconstruction minus its paired reference; the deviation index is its mean absolute value across features.

See [Python examples](docs/usage.md) for model setup, checkpoint loading, and inference. Reuse training preprocessing and feature order for new samples.

## Citation

If you use this code, please cite:

```bibtex
@misc{chintapalli2026casl,
  title={CASL-VAE: Learning Structured Latent Variables from Unpaired Data for Semi-supervised Clustering and Paired Sample Generation},
  author={Chintapalli, Sai Spandana and Chaudhari, Pratik and Davatzikos, Christos},
  journal={arXiv preprint arXiv:2607.08254},
  year={2026},
}
```

## License

Copyright (c) 2016 University of Pennsylvania. All rights reserved. See [https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html](https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html)

## Contact

Sai Spandana Chintapalli — [schinta@engineering.upenn.edu](mailto:schinta@engineering.upenn.edu)
