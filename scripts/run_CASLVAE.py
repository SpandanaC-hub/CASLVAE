"""Train CASLVAE on unpaired reference and target CSV data."""
from pathlib import Path
import argparse
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from CASLVAE_model import CASLVAE
from train_utils import train_model
from data_utils import CreateDataset, PreprocessingStats, PreprocessingTransform
from torch.utils.data import DataLoader
from scipy.special import softmax
import seaborn as sns
print(ROOT)

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


def run_training(ncategories, DEVICE, epochs, max_attempts, data_file, output_dir, data_options=None):
    """Run with explicit model options and separate CSV/preprocessing settings."""
    data_options = dict(data_options or {})
    seed = data_options.get("seed", 42)
    covariate_correct = data_options.get("covariate_correct", False)
    output_dir = Path(output_dir)
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

     # load data and aplit into train, cross-val, test
    development, test, features, covariates = load_development_data(data_file, **data_options)
    if data_options.get("use_csv_splits", True) and "split" in development:
        train = development[development["split"].eq("train")].copy()
        validation = development[development["split"].eq("validation")].copy()
    else:
        train, validation = train_test_split(
            development, test_size=data_options.get("validation_fraction", .1),
            random_state=seed + 1, stratify=development["label"])

    for name, frame in (("train", train), ("validation", validation)):
        if set(frame["label"]) != {"Background", "Target"}:
            raise ValueError(name + " must contain reference and target samples")
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.to_csv(output_dir / (name + "_split.csv"), index=False)

    # get covariate correction parameters if covariate_correct is True
    stats = PreprocessingStats(
        train[features].to_numpy(dtype=float), train["label"].to_numpy(),
        train[covariates].to_numpy(dtype=float), covariate_correct=covariate_correct)
    stats.stats()

    # preprocess the data based on the training data stats
    def transform(frame):
        return PreprocessingTransform(
            frame[features].to_numpy(dtype=float), stats,
            frame[covariates].to_numpy(dtype=float),
            covariate_correct=covariate_correct).transform().astype(np.float32)

    train_dataset = CreateDataset(transform(train), train["label"].to_numpy())
    cv_dataset = CreateDataset(transform(validation), validation["label"].to_numpy())

    # define model hyperparameters
    opt = default_options(nfeatures=len(features), ncategories=ncategories, path=str(output_dir), epochs=epochs, DEVICE=DEVICE)
    train_dataloader = DataLoader(train_dataset, batch_size=opt["batch_size"], shuffle=True)
    data_params = dict(
        data_std=stats.data_std, data_mean=stats.data_mean,
        cov_min=stats.cov_min, cov_max=stats.cov_max, cov_beta=stats.cov_beta,
        covariate_correct=covariate_correct,
        feature_names=features, covariate_names=covariates, tau=1.)
    with (output_dir / "dataparams_real_dict.pkl").open("wb") as stream:
        pickle.dump(data_params, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (output_dir / "options.json").write_text(json.dumps(opt, indent=2))

    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float32, device=opt["DEVICE"])

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    converged = False
    for attempt in range(1, max_attempts + 1):
        model = CASLVAE()
        model.create(opt=opt)
        model.set_preprocessing_params(data_params)
        model, clusteringM, loss_list, converged = train_model(
            model, opt, train_dataloader,
            tensor(train_dataset.data_bg), tensor(train_dataset.data_tgt),
            tensor(cv_dataset.data_bg), tensor(cv_dataset.data_tgt))
        print("Training attempt:", attempt, "| Converged:", converged)
        if converged:
            break
    # the attempt that supplies the saved result.
    (output_dir / "training_status.json").write_text(json.dumps(
        dict(attempts=attempt, converged=bool(converged)), indent=2))

    scat_prob = softmax(clusteringM[0].reshape(-1, opt["ncategories"]), axis=-1) # get salient category membership
    # save the training curves
    plt.figure(figsize=(35,10))
    plt.subplot(2,7,1)
    plt.plot(loss_list['recon_x'][0])
    plt.plot(loss_list['recon_x'][1])
    plt.plot(loss_list['recon_y'][0])
    plt.plot(loss_list['recon_y'][1])
    plt.legend(['train MSE tgt','val MSE tgt','train MSE ref','val MSE ref'])
    
    plt.subplot(2,7,2)
    plt.plot(loss_list['w_list_tgt'][0])
    plt.plot(loss_list['w_list_tgt'][1])
    plt.plot(loss_list['w_list_ref'][0])
    plt.plot(loss_list['w_list_ref'][1])
    plt.legend(['train WD tgt','val WD tgt','train WD ref','val WD ref'])
    
    plt.subplot(2,7,3)
    plt.plot(loss_list['w_list_unpair'][0])
    plt.plot(loss_list['w_list_unpair'][1])
    plt.legend(['real tgt-ref WD','synth tgt-ref WD'])
    
    plt.subplot(2,7,4)
    plt.plot(loss_list['kl_loss'][0])
    plt.title('kl z')
    plt.subplot(2,7,5)
    plt.plot(loss_list['kl_loss'][1])
    plt.title('kl scat x')
    plt.subplot(2,7,6)
    plt.plot(loss_list['kl_loss'][2])
    plt.title('kl scont x')
    
    plt.subplot(2,7,7)
    plt.plot(loss_list['mmd_loss'])
    plt.title('mmd loss')
    
    plt.subplot(2,7,8)
    plt.plot(loss_list['tau'])
    plt.title('gumbel temperature tau')
    
    plt.subplot(2,7,9)
    plt.plot(loss_list['entropy'])
    plt.title('BCE penalty on scat')
    
    plt.subplot(2,7,10)
    plt.plot(loss_list['norm_reconsalient_y'])
    plt.title('L1 norm of ref domain salient latent recon')
    
    plt.subplot(2,7,11)
    plt.plot(loss_list['consistency_list'])
    plt.title('latent consistency')
    
    plt.subplot(2,7,12)
    plt.plot(loss_list['kl_loss'][3])
    plt.title('kl scont y penalty')
    
    plt.subplot(2,7,13)
    plt.plot(loss_list['kl_loss'][4])
    plt.title('kl scontprior y penalty')
    
    plt.subplot(2,7,14)
    sns.scatterplot(x = scat_prob[:,-2], y=scat_prob[:,-1], hue=cv_dataset.tgt_labels)
    
    plt.xlabel("Second-last category probability")
    plt.ylabel("Last category probability")
    plt.title('scat probabilities')
    plt.savefig(output_dir / "monitor_loss_0.jpg")
    plt.close()
    

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "caslvae_training_output",
                        help="Directory for checkpoints and plots.")
    parser.add_argument("--ncategories", type=int, default=3,
                        help="Number of categories, including reference category 0.")
    parser.add_argument("--data-file", type=Path, default=ROOT / "docs/synthetic_data/synthetic_cross_sectional.csv",
                        help="path to CSV file with data")
    parser.add_argument("--feature-names", type=Path, default=ROOT / "docs/synthetic_data/synthetic_featurenames.csv",
                        help="path to CSV file with feature names")
    parser.add_argument("--covariate-correct", action="store_true", 
                        help="Argument enables covariate correction using the specified covariates.")
    parser.add_argument("--covariates", nargs="*", default=[],
                        help="Numeric, nonconstant covariate columns, e.g. Age Sex ICV.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=3, help="Number of attempts to train model until convergence")
    args = parser.parse_args()
    if args.ncategories < 2:
        parser.error("ncategories must be at least 2.")
    if args.covariate_correct and not args.covariates:
        parser.error("--covariate-correct requires --covariates.")
    if args.epochs < 0:
        parser.error("epochs must be graeter than 0.")

    data_options = dict(
        featurenames_file=args.feature_names,
        covariate_correct=args.covariate_correct,
        covariates=args.covariates,
        test_fraction=0.20,
    )

    DEVICE="cuda" if torch.cuda.is_available() else "cpu"

    run_training(ncategories = args.ncategories, epochs = args.epochs, max_attempts = args.max_attempts,  DEVICE = DEVICE, data_file = args.data_file, output_dir = args.output, data_options=data_options)


if __name__ == "__main__":
    main()

