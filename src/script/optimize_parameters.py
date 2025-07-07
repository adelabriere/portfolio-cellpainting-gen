from torchvision.transforms import v2
import torch
import os
import sys
import argparse
import json

# We add the correct path to the system

SCRIPT_PATH = os.path.realpath(__file__)
MODULE_PATH = os.path.abspath(os.path.join("src"))
sys.path.append(MODULE_PATH)


import gencellpainting.evaluation.hparams_optimization as hopt
from gencellpainting.model import *
from gencellpainting.utils.dataset import CellPaintingDatasetInMemory, WGANCriticDataset

FIXED_PARAMS = "fixed"
GRID_PARAMS = "grid_search"
TRAINER_PARAMS = "trainer"
NUM_WORKERS = 4

def get_model(model_type):
    if model_type == "VAE":
        return VAE
    if model_type == "WGAN_GP":
        return WGAN_GP
    if model_type == "diffusion":
        return DiffusionProcess
    else:
        raise KeyError(f"Model type not recognized {model_type}")
    
def collate_wgan_batch(batch):
    gen_imgs,disc_imgs = zip(*batch)
    gen_imgs = torch.stack(gen_imgs)
    disc_imgs = torch.stack([y for subbatch in disc_imgs for y in subbatch ])
    return gen_imgs, disc_imgs
    
def get_transforms():
    transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True) # This also set the values of the tensor between 0 and 1
    ])
    return transforms

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def load_data(path_tensor, transforms, test_frac=0.2, batch_size=64, is_gan=False):
    ds = CellPaintingDatasetInMemory(tensor=torch.load(path_tensor))
    ds.transform = transforms

    if is_gan:
        ds = WGANCriticDataset(ds,ncritic=5)

    ds_train, ds_test = torch.utils.data.random_split(ds, [1-test_frac,test_frac])

    supp_args = {
        "batch_size":batch_size,
        "shuffle":True,
        "drop_last":True,
        "num_workers":NUM_WORKERS
    }

    if is_gan:
        supp_args["collate_fn"] = collate_wgan_batch

    # Creating the dataloader
    dl_train = torch.utils.data.DataLoader(ds_train, **supp_args)
    dl_test = torch.utils.data.DataLoader(ds_test, **supp_args)
    return dl_train, dl_test

def check_args(args):
    if not os.path.exists(args.tensorfile):
        raise ValueError(f"The tensor file {args.tensorfile} does not exist.")
    if not os.path.exists(args.config):
        raise ValueError(f"The config file {args.config} does not exist.")
    config = load_config(args.config)
    if not all(key in config for key in [FIXED_PARAMS, GRID_PARAMS]):
        raise ValueError(f"The config file {args.config} does not contain the required keys: {FIXED_PARAMS}, {GRID_PARAMS}.")
    

def optimize_parameters(args):

    config = load_config(args.config)
    batch_size = int(config["batch_size"])

    transforms = get_transforms()
    dl_train, dl_test = load_data(args.tensorfile, transforms=transforms,
                                  batch_size=batch_size, is_gan=config["is_gan"])
    
    model_constructor = get_model(config["model_type"])

    # Collecting differnet arguments
    args_to_test = config[GRID_PARAMS]
    args_fixed = config[FIXED_PARAMS]
    args_trainer = config[TRAINER_PARAMS]

    complete_folder = os.path.join(args.folder, config["name_model"])

    # Defining the optimizer
    grid_search = hopt.HyperParametersOptimizer(model=model_constructor, dl=dl_train, dl_val=dl_test,\
                                            hparams=args_to_test, fixed_args=args_fixed,metric=config["metric"],
                                            early_stopping_metric=config["metric"], trainer_args=args_trainer,
                                              name_model=config["name_model"], folder=complete_folder)
    
    print(f"Starting parameters optimization, on a total of {grid_search.nparams} parameters set")
    _ = grid_search.optimize_hyperparameters()
    best_params = grid_search.get_best_params()


    params_paths = write_params(complete_folder, args.name, best_params)
    return params_paths

def write_params(folder, name, params):
    path_params = os.path.join(folder, "best_parameters_"+name+".json")
    with open(path_params, "w") as f:
        json.dump(params, f)
    return path_params

if __name__ == "__main__":
    # Creating the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Store the results of the optimization.",
                    type=str)
    parser.add_argument("--name", help="Name of the model.",
                    type=str)
    parser.add_argument("--tensorfile", help="File of the dataset/tensor to use as dataset for optimization.",
                    type=str)
    parser.add_argument("--config", help="The config json file storing the parameters")
    
    args = parser.parse_args()

    check_args(args)
    best_params = optimize_parameters(args)
    params_paths = write_params(args.folder, args.name, best_params)

    print("Parameters have been written in {}".format(params_paths))

# python src/script/optimize_parameters.py --folder "data/optim" --name "VAE_script" --tensorfile "/mnt/c/Users/alexi/Documents/data/images/cellpainting/cpg0016-jump/data/jump_128px_uint8.pt" --config "data/optim/config/VAE_config.json"




