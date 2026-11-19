# CASL-VAE usage examples

Run these examples from the repository root.

For the script-based workflow, category selection, and experiment reporting, see the [experiment guide](experiments.md).

For a complete synthetic example with data and latent-space plots, open [the cross-sectional 145-ROI notebook](CASLVAE_synthetic_145ROIs.ipynb). It imports the original model from `code/` and runs without covariate correction on a completely synthetic unpaired reference-target dataset.

## Prepare data and instantiate

Run from the repository root, including when using a notebook:

```python
import sys
sys.path.insert(0, "code")

import pandas as pd
import torch
from torch.utils.data import DataLoader
from CASLVAE_model import CASLVAE
from data_utils import CreateDataset, train_test_split_unpaired

df = pd.read_csv("data/your_data.csv")
features = [column for column in df.columns if column.startswith("ROI_")] # or supply dataset specific feature names
train_x, train_labels, _, val_x, val_labels, _, stats = train_test_split_unpaired(
    df, feature_cols=features, label_col="label", covariate_cols=[],
    test_size=0.2, stratify=True, random_state=42, covariate_correct=False,
)
# Normalization uses training Background statistics for both splits.

opt = dict(
    nfeatures=len(features), nLatent_z=15, nLatent_s_cont=2,
    nLatent_s_cat=1, ncategories=3,  # reference + two target categories
    kl_z_lambda=1, kl_scat_lambda=1, kl_scont_lambda=1,
    kl_sconty_lambda=1, mmd_lambda=100, consistency_lambda=10,
    entropy_lambda=1, norm_salienty_lambda=0,
    lr=1e-3, beta1=0.5, scheduler=False,
    DEVICE="cuda" if torch.cuda.is_available() else "cpu",
    epochs=500, batch_size=128, path="outputs",
)
model = CASLVAE()
model.create(opt)
model.set_preprocessing_params(dict(
    data_mean=stats.data_mean, data_std=stats.data_std,
    cov_min=stats.cov_min, cov_max=stats.cov_max, cov_beta=stats.cov_beta,
    covariate_correct=stats.covariate_correct,
    feature_names=features, covariate_names=[], tau=1.0,
))
loader = DataLoader(CreateDataset(train_x, train_labels),
                    batch_size=opt["batch_size"], shuffle=True)
reference, target = next(iter(loader))
losses = model.train_instance(
    target.to(opt["DEVICE"], dtype=torch.float32),
    reference.to(opt["DEVICE"], dtype=torch.float32), tau=1.0,
)
```

These coefficients are an example configuration, not universal defaults. Adam uses weight decay `0.0025`; `entropy_lambda` weights the supervised reference/target BCE-style loss. Use `nLatent_s_cat=1` with the reference-preset implementation. The example performs **one training update**. The supported examples use `scheduler=False`.

## Run the full training loop

After preparing the data and model above:

```python
from pathlib import Path
from train_utils import train_model

Path(opt["path"]).mkdir(parents=True, exist_ok=True)
validation = CreateDataset(val_x, val_labels)

def tensor(values):
    return torch.as_tensor(values, dtype=torch.float32, device=opt["DEVICE"])

model, clustering_logits, loss_history, converged = train_model(
    model, opt, loader,
    tensor(loader.dataset.data_bg), tensor(loader.dataset.data_tgt),
    tensor(validation.data_bg), tensor(validation.data_tgt),
)
```

The single-update example above is optional; skip it when starting a fresh full training run. Here the held-out split is **validation**, used for monitoring and stopping. Reserve a separate test set for final evaluation. The checkpoint is saved when the original stopping rule triggers or the epoch budget is exhausted; it is not a minimum-BCE checkpoint. See [training behavior](experiments.md#training-and-checkpoint-selection).

## Access outputs

Use the model from the example above, or load a trained checkpoint:

```python
model = CASLVAE()
checkpoint = model.load_model("outputs/model_checkpoint.pt", device=opt["DEVICE"])
```

For a loaded model, preprocess new data with the saved feature order, normalization, and covariate-correction parameters. Do not refit preprocessing on the test set. The example below assumes `val_x` already uses the model's training preprocessing.

This deterministic example uses posterior means and the most likely category, following the paper's hard category assignment at inference:

```python
g = model.G_forward
g.eval()
x = torch.as_tensor(val_x[:8], dtype=torch.float32, device=opt["DEVICE"])
with torch.no_grad():
    z, _ = g.encode_common(x)
    logits, features_s = g.encode_salient_cat(x)
    probabilities = logits.softmax(-1)
    category = probabilities.argmax(-1)
    category_code = torch.nn.functional.one_hot(
        category, num_classes=probabilities.shape[-1]
    ).to(x.dtype)
    s = g.enc_s_cont(torch.cat([features_s, category_code], dim=-1))
    reference_code = torch.zeros_like(probabilities)
    reference_code[:, 0] = 1
    common = g.decode_common(z)
    reconstruction = common + g.decode_salient(category_code, s)
    counterpart = common + g.decode_salient(reference_code, torch.zeros_like(s))
    salient_deviation = reconstruction - counterpart
    deviation_index = salient_deviation.abs().mean(dim=1)
```


Outputs are standardized. Invert normalization with `value * (model.train_std + 1e-6) + model.train_mean`; multiply differences by `model.train_std + 1e-6` only. When covariate correction was enabled, inverted reconstructions remain in covariate-corrected units; add the saved covariate effects back to recover original feature units. DI above is computed in standardized feature space. Counterparts are model estimates rather than observed individual healthy anatomy.

For prior-based paired generation, sample a common `z`, choose a target one-hot category code, and sample `s` around `g.prior_s_cont(category_code)`. Decode the same `z` with the reference preset to obtain its paired reference. Independent `z` draws generate unpaired samples.

### Files saved by the research scripts

| File | Contents |
|---|---|
| `model_checkpoint.pt` | Network and optimizer states, options, preprocessing parameters, feature/covariate names, temperature, epoch, and losses |
| `dataparams_real_dict.pkl` | Preprocessing parameters saved by the real-data training script |
| `test_split.csv` | Held-out rows saved before training in the real-data script |
| `monitor_loss_0.jpg` | Training/validation monitoring plot |
| `CASLVAE_quality_epoch*.jpg` | Feature-space reconstruction checks from the epoch loop |
| `pairwise_ari_dict.pkl`, `pairwiseari.jpg` | Stability scores and plot for candidate category counts |
| `clustering_results.pkl` | Logits from the repeated stability runs |

Inference tensors are computed in Python; the current scripts do not provide a general-purpose CSV inference command.


