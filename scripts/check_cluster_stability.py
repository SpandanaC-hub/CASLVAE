"""Configurable copy of scripts/check_cluster_stability.py.

Run: python scripts/check_cluster_stability.py --help
Stability uses training data only to assess the reproducibility of the selected salient categories M
"""
from pathlib import Path
import sys
import argparse
import json
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from cluster_utils import crossval_cluster_stability, pairwise_ari

# Share CSV preparation and model defaults with the training runner.
def load_development_data(data_file, featurenames_file, label_column="label",
                          reference_label="Background", subject_column="subject_id",
                          covariate_correct=False, covariates=None, seed=42,
                          test_fraction=.20, validation_fraction=.1,
                          use_csv_splits=True):
    data = pd.read_csv(data_file).reset_index(drop=True)
    if label_column not in data:
        raise ValueError("Missing label column: " + label_column)
    if data[label_column].isna().any():
        raise ValueError("Labels must not be missing")
    data["label"] = np.where(data[label_column].eq(reference_label), "Background", "Target")
    features = pd.read_csv(featurenames_file)['feature_names'].tolist()
    covariates = (covariates or []) if covariate_correct else []
    if len(features) < 8:
        raise ValueError("Specify at least 8 features for the existing training monitors")
    if covariate_correct and not covariates:
        raise ValueError("Specify covariates when covariate correction is enabled")
    for columns in (features, covariates):
        if columns and not np.isfinite(data[columns].to_numpy(dtype=float)).all():
            raise ValueError("Features/covariates must be numeric and finite")
    if subject_column and data[subject_column].duplicated().any():
        raise ValueError("These cross-sectional templates expect one row per subject")
    if use_csv_splits and "split" in data:
        if not data["split"].isin(["train", "validation", "test"]).all():
            raise ValueError("split must contain only train, validation, or test")
        development = data[data["split"].ne("test")].copy()
        test = data[data["split"].eq("test")].copy()
    else:
        development, test = train_test_split(
            data, test_size=test_fraction, random_state=seed, stratify=data["label"])
    for name, frame in (("development", development), ("test", test)):
        if set(frame["label"]) != {"Background", "Target"}:
            raise ValueError(name + " must contain reference and target samples")
    return development.reset_index(drop=True), test.reset_index(drop=True), features, covariates


def default_options(nfeatures, ncategories, nLatent_z=15, nLatent_s_cont=2, nLatent_s_cat=1,
        mmd_lambda=100, kl_scat_lambda=1, kl_scont_lambda=1,
        kl_sconty_lambda=1, kl_z_lambda=1, consistency_lambda=10,
        norm_salienty_lambda=0, entropy_lambda=1,
        lr=.001, beta1=.5, DEVICE='cpu', epochs=500,
        decay_epochs=1, scheduler=False, batch_size=128, path='./'):
    """Model settings; training supplies the feature count and output path."""
    return dict(nfeatures=nfeatures,# number of features
        nLatent_z=nLatent_z, # common latent dimension
        nLatent_s_cont=nLatent_s_cont, # salient continuous latent dimension
        nLatent_s_cat=nLatent_s_cat, 
        ncategories=ncategories, # number of salient categories including reference category index 0
        mmd_lambda= mmd_lambda, # hyperparameter to control mmd penalty
        kl_scat_lambda=kl_scat_lambda, # hyperparameter to control kl on s cat for target
        kl_scont_lambda=kl_scont_lambda, # hyperparameter to control kl on s cont for target 
        kl_sconty_lambda=kl_sconty_lambda, # hyperparameter to control penalty on s cont for reference
        kl_z_lambda=kl_z_lambda, # hyperparameter to control kl on common latent z
        consistency_lambda=consistency_lambda,# hyperparameter to control latent consistency of z
        norm_salienty_lambda=norm_salienty_lambda, # OPTIONAL TODO: hyperparameter to control L1 norm penalty on salient reconstruction for reference
        entropy_lambda=entropy_lambda,# hyperparameter to control BCE penalty on preset reference category
        lr=lr, # learning rate
        beta1=beta1,  #beta1 hyperparameter in Adam optimizer
        DEVICE=DEVICE, # use 'cpu' or 'cuda'
        epochs=epochs,
        decay_epochs=decay_epochs, # TODO: currently diabled, epoch number when learning rate decay should begin
        scheduler=scheduler,  #True to set learning rate scheduler
        batch_size=batch_size, 
        path=path # path of directory to save model
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_stability(opt, data_file, output_dir, data_options=None, m_list=None):
    """Run with explicit model options and separate CSV/preprocessing settings."""
    opt = dict(opt)  # updates ncategories and other options in-place.
    data_options = dict(data_options or {})
    seed = data_options.get("seed", 42)
    covariate_correct = data_options.get("covariate_correct", False)
    output_dir = Path(output_dir)
    set_seed(seed)
    m_list = list([2, 3, 4, 5, 6] if m_list is None else m_list)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, test, features, covariates = load_development_data(data_file, **data_options)
    dataset.to_csv(output_dir / "train_split.csv", index=False)
    test.to_csv(output_dir / "test_split.csv", index=False)
    opt.update(nfeatures=len(features), path=str(output_dir))
    (output_dir / "options.json").write_text(json.dumps(
        dict(opt=opt, candidate_categories=m_list), indent=2))
    # It monitors all samples, including those in each training subset.
    # Its optional covariate correction is fitted before repeated subsampling.
    clustering_logits = crossval_cluster_stability(
        dataset.copy(), opt, covariate_correct, features, covariates, m_list)
    results = {m: pairwise_ari(logits) for m, logits in zip(m_list, clustering_logits)}
    with (output_dir / "pairwise_ari_dict.pkl").open("wb") as stream:
        pickle.dump(results, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with (output_dir / "clustering_results.pkl").open("wb") as stream:
        pickle.dump(dict(logits=clustering_logits), stream, protocol=pickle.HIGHEST_PROTOCOL)
    fig, ax = plt.subplots()
    ax.boxplot([results[m] for m in m_list], showmeans=True)
    ax.set(xticks=np.arange(1, len(m_list)+1), xticklabels=m_list,
           xlabel="Number of categories M (including reference)",
           ylabel="Pairwise ARI across 10 repeated training subsets")
    fig.tight_layout()
    fig.savefig(output_dir / "pairwiseari.jpg", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "caslvae_cluster_stability_output",
                        help="Directory for stability results and plots.")
    parser.add_argument("--data-file", type=Path,
                        default=ROOT / "docs/synthetic_data/synthetic_cross_sectional.csv")
    parser.add_argument("--feature-names", type=Path,
                        default=ROOT / "docs/synthetic_data/synthetic_featurenames.csv")
    parser.add_argument("--m-list", type=int, nargs="+", default=[2, 3, 4, 5, 6],
                        help="Candidate category counts, each including reference category 0.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--covariate-correct", action="store_true", help="Argument enables covariate correction using the specified covariates.")
    parser.add_argument("--covariates", nargs="*", default=[],
                        help="Numeric, nonconstant covariate columns, e.g. Age Sex ICV.")
    parser.add_argument("--device", type=str, default=None, help="select cpu or cuda")

    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be greater than 0.")
    if any(m < 2 for m in args.m_list) or len(set(args.m_list)) != len(args.m_list):
        parser.error("m-list must contain distinct category counts of at least 2.")
    if args.covariate_correct and not args.covariates:
        parser.error("--covariate-correct requires --covariates.")
    if args.device is None:
      args.device = "cuda" if torch.cuda.is_available() else "cpu"
      
    data_options = dict(
        featurenames_file=args.feature_names,
        covariate_correct=args.covariate_correct, covariates=args.covariates,
        test_fraction=.30)
    opt = default_options(
        nfeatures=0, ncategories=args.m_list[0], epochs=args.epochs,
        DEVICE=args.device)
    run_stability(opt, args.data_file, args.output, data_options, m_list=args.m_list)


if __name__ == "__main__":
    main()

