# Running CASL-VAE experiments

This guide covers the cross-sectional unpaired input CASL-VAE implementation.

The current scripts contain dataset-specific preparation and private paths. They must be configured before use; they are not general-purpose CSV command-line interfaces.

## Installation

From the repository root:

```bash
conda env create -f environment.yml
conda activate caslvae
```

Use the same environment for training and inference. Set `opt['DEVICE']` to `cuda` when available, otherwise `cpu`. The examples use `scheduler=False`.

The research scripts also import `pickle5`, which is not listed in `environment.yml`, and include local Python environment paths. Resolve these script-specific imports/paths before running. The environment file alone does not make those templates portable.

## Data preparation

Supply one row per sample, numeric feature columns, and a population label:

```csv
subject_id,label,ROI_1,ROI_2,Age,Sex,ICV
sub-001,Background,3250,6100,65,0,1450000
sub-002,Target,2900,5700,68,1,1430000
```

- The reference label must be exactly `Background`. All other labels are treated as target.
- Both populations must occur in training and validation. Subtype labels and matched pairs are not required.
- Specify an ordered feature list; exclude identifiers, labels, and metadata from the feature matrix.
- Resolve missing/non-numeric features before training. Numerically encode categorical covariates consistently.
- For repeated observations, split by subject so one individual's scans cannot cross partitions. The supplied cross-sectional split helper splits rows, not subjects.

`CreateDataset` returns unpaired reference/target samples, cycling the smaller population to match the larger population's length. These minibatch combinations are not known biological pairs.

### Preprocessing

With `covariate_correct=False`, set `covariates_list=[]`. Feature means and standard deviations are estimated from training reference samples and applied to both populations.

With `covariate_correct=True`, provide nonconstant numeric covariates. Their scaling and feature regression coefficients are fitted on training reference samples, followed by normalization of corrected reference features. Save and reuse these parameters for validation, test, and inference.

This is covariate correction before CASL-VAE fitting using the train set. Keep feature/covariate ordering fixed. Do not fit preprocessing on the test set.

## Configure the existing runner

[`scripts/run_CASLVAE_realdata.py`](../scripts/run_CASLVAE_realdata.py) follows the same basic workflow as the simpler [`code/train.py`](../code/train.py), with additional data preparation, retries, and plotting. Use the scripts runner as the experiment entry point once configured. The minimal template still has stale model names and preprocessing assignments; [usage examples](usage.md) show the current API.

Before running, configure:

| Setting | What to supply |
|---|---|
| Data-loading block | Local data source; remove assumptions specific to the example cohort |
| Group definition | Which population maps to `Background` |
| `brain_measures_list` | Ordered feature names |
| `covariates_list`, `covariate_correct` | Optional preprocessing covariates |
| `opt` | Model dimensions, losses, training settings, device |
| `opt['path']` | Separate output directory for each experiment |
| Splits and seeds | Reproducible development and held-out test subjects |

The current real-data runner first reserves 30% of rows as `test_split.csv`, then uses a 1% validation split within the remaining rows. This is the current script behavior, not a recommended universal split ratio. Its commented ten-fold block is not the active training workflow. The runner can restart training up to three times when the convergence flag is false.

## Model options

These are example settings consistent with the main runner; category count depends on the experiment.

| Option | Example | Meaning |
|---|---:|---|
| `nfeatures` | Number of selected features | Input/output dimension |
| `nLatent_z` | 15 | Common latent dimension |
| `nLatent_s_cont` | 2 | Within-category salient dimension |
| `nLatent_s_cat` | 1 | Number of categorical variables; keep 1 |
| `ncategories` | 3 | Total categories, including reference index 0 |
| `kl_z_lambda` | 1 | Common latent KL weight |
| `kl_scat_lambda` | 1 | Categorical latent KL weight |
| `kl_scont_lambda` | 1 | Target continuous salient KL weight |
| `kl_sconty_lambda` | 1 | Reference continuous salient penalty weight |
| `mmd_lambda` | 100 | Common-distribution matching weight |
| `consistency_lambda` | 10 | Latent consistency weight |
| `entropy_lambda` | 1 | Reference/target BCE-style supervision weight |
| `norm_salienty_lambda` | 0 | Optional reference salient-reconstruction penalty |
| `lr`, `beta1` | 0.001, 0.5 | Adam settings; second beta is 0.999 |
| `batch_size`, `epochs` | 128, 500 | Batch size and maximum training epochs |
| `scheduler` | False | Configuration used in this guide |
| `DEVICE`, `path` | Device, output directory | Execution and output settings |

Adam's weight decay is fixed at `0.0025` in `code/CASLVAE_model.py`, rather than supplied through `opt`. `ncategories=3` means one reference and two target categories. Do not interpret categorical indices as ordered disease severity.

## Select the number of categories

Configure the data/feature definitions and `M_list` in [`scripts/check_cluster_stability_realdata.py`](../scripts/check_cluster_stability_realdata.py), then run:

```bash
python scripts/check_cluster_stability_realdata.py
```

The current template has an active dataset load outside its commented feature-definition block; define the feature list explicitly before running.

`crossval_cluster_stability` uses ten repeated 80% training subsets. Models assign categories to the same full target population, and pairwise ARI measures agreement between assignments. This is **resampling stability**, not held-out accuracy: the monitoring population includes training subjects. When covariate correction is enabled, the current utility fits that correction before the repeated splits.

Inspect stability across candidate category counts together with category sizes and reconstruction behavior. Stability alone does not establish the biological correctness of a category count. Perform selection on development data and reserve independent test subjects for final evaluation.

## Training and checkpoint selection

After configuration:

```bash
python scripts/run_CASLVAE_realdata.py
```

Alternatively, the [Python usage guide](usage.md#run-the-full-training-loop) calls the same training loop directly without the cohort-specific preparation.

Gumbel temperature anneals from 1 toward 0.3. The original stopping check runs every five epochs and requires all of the following:

- Temperature at or below approximately 0.5 and more than half the maximum epoch budget completed.
- Stable target-category assignments over the recent checks.
- Both reference and target validation reconstruction losses exceeding their recent minima by the configured tolerance.

The loop saves the model at this stopping point, or at the final epoch if the rule does not trigger. It does **not** automatically restore minimum validation reconstruction or minimum BCE weights. The checkpoint's `epoch`, `tau`, and loss history identify what was saved. The convergence flag reports whether the original stopping rule triggered.

## Outputs and inference

| Output | Contents |
|---|---|
| `model_checkpoint.pt` | Network/optimizer states, options, preprocessing, feature names, epoch, temperature, losses |
| `dataparams_real_dict.pkl` | Runner's preprocessing metadata |
| `test_split.csv` | Reserved test rows from the real-data runner |
| `monitor_loss_0.jpg` | Runner's training/validation monitoring figure |
| `CASLVAE_quality_epoch*.jpg` | Feature-space reconstruction monitors |
| `pairwise_ari_dict.pkl`, `pairwiseari.jpg` | Category-count stability results |
| `clustering_results.pkl` | Category logits from stability runs |

See [inference examples](usage.md#access-outputs) for probabilities, latent embeddings, reconstruction, paired reference estimation, and deviation indices. To generate paired samples, hold common `z` fixed while changing salient codes. Paired reference estimates are model-derived counterparts, not observed individual reference anatomy.

## Reporting and paper reproduction

For each experiment, record the dataset/cohort criteria, ordered features, splits, seeds, preprocessing, complete `opt`, selected category count, saved epoch/temperature, and number of training attempts.

For semi-synthetic experiments, additionally record the source reference cohort, affected-feature masks, perturbation strengths, subtype proportions, and the retained clean counterparts. Report target-only ARI separately from ARI over reference plus target samples. Distinguish ground-truth ARI from resampling stability. State the units and feature scope of reconstruction/deviation metrics.

The project page summarizes paper results, including semi-synthetic metrics averaged over ten repeated splits. The two scripts above cover real-data training and category stability; they do not by themselves reproduce every paper table or generate the semi-synthetic datasets. Exact reproduction also requires the corresponding data preparation, experiment configuration, and evaluation procedure. Private cohort paths are not downloadable example data.

The [project page](../project-page/README.md) provides the scientific overview. This guide and the usage examples describe execution; neither changes the original model or introduces the longitudinal sandbox methods.
