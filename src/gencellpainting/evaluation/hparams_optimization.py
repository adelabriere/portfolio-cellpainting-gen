import logging
import os
import pandas as pd

from torch.utils.data import DataLoader
from typing import Optional
from itertools import product
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from gencellpainting.model.abc_model import UnsupervisedImageGenerator
from gencellpainting.constants import MONITORED_LOSS

class HyperParametersOptimizer:
    def __init__(self, model: type, dl: DataLoader, hparams: dict,\
                 folder:str, metric:str=MONITORED_LOSS,early_stopping_metric:str = "train_loss",\
                dl_val: Optional[DataLoader]=None, patience:int = 5,\
                      name_model:str = "model", trainer_args = {"accelerator":"gpu", "max_epochs":100},\
                 fixed_args=None):        
        self.hparams = hparams# Model hyperparamters as a dictionnary
        self.fixed_args = fixed_args
        if self.fixed_args is None:
            self.fixed_args = {}
        self.model = model
        self.dl = dl
        self.dl_val = dl_val
        self.metric = metric
        self.folder = folder
        self.folder_log = None
        self.folder_model = None
        self.csv_path = None
        self.name_model = name_model
        self.patience = patience
        self.early_stopping_metric = early_stopping_metric
        self.trainer_args = trainer_args

        self._initialize()

    def _initialize(self):
        nparams = 1
        for v in self.hparams.values():
            nparams = nparams * len(v)
        self.nparams = nparams
        self._initialize_paths()
        
    def _initialize_paths(self):
        
        # Creating the output folder
        dirname = os.path.dirname(self.folder)
        if not os.path.exists(dirname):
            raise FileNotFoundError("Parent folder {dirname} needs to exist")
        
        # Creating the folders
        os.makedirs(self.folder, exist_ok = True)

        path_logs = os.path.join(self.folder, "logs")
        if not os.path.isdir(path_logs):
            os.makedirs(path_logs, exist_ok=True)
        self.folder_log = path_logs

        path_models = os.path.join(self.folder, "models")

        os.makedirs(path_models, exist_ok=True)
        self.folder_model = path_models

        self.csv_path = os.path.join(self.folder, "metrics.csv")

    def build_name(self,params):
        return "_".join([self.name_model]+[(str(k)[0])+str(v) for k,v in params.items()])

    def single_run(self, hparams):

        # Instantiate the model
        cmodel = self.model(**hparams, **self.fixed_args)

        model_name = self.build_name(hparams)

        # Creating the logger
        tb_logger = L.pytorch.loggers.TensorBoardLogger(self.folder_log, name=model_name)

        filename_fmt = '{'+model_name+'}'
        callbacks = []

        if self.early_stopping_metric is not None:
            # Callbacks
            early_stopping = EarlyStopping(\
                monitor=self.early_stopping_metric,mode="min", patience=self.patience)
            filename_fmt = filename_fmt+'-{'+self.early_stopping_metric+'}'
            callbacks.append(early_stopping)

        checkpoint_callback = ModelCheckpoint(
            monitor=None, # Save the last weights only
            dirpath=self.folder_model,filename=filename_fmt)
        callbacks.append(checkpoint_callback)

        # Trainer
        trainer = L.Trainer(logger=tb_logger, callbacks=callbacks,\
                            enable_progress_bar=False, **self.trainer_args)
        
        dl_args = {"train_dataloaders": self.dl}
        if self.dl_val is not None:
            dl_args["val_dataloaders"] = self.dl_val
        
        # Training
        trainer.fit(cmodel, **dl_args)

        # Extracting all the metrics values from the model
        dic_metrics = {k:float(v) for k,v in trainer.logged_metrics.items()}
        return dic_metrics
    
    def log_metrics(self, i, hparams, metrics):
        if i==0:
            with open(self.csv_path, "w") as csv:
                # Writing the header
                names_hparams = [f"hparam/{k}" for k in hparams.keys()]
                headers = names_hparams + list(metrics.keys())
                csv.write(",".join(headers)+"\n")
        with open(self.csv_path, "a") as csv:
            # Writing the values
            values = list(hparams.values()) + list(metrics.values())
            values = [str(v) for v in values]
            csv.write(",".join(values)+"\n")

    def optimize_hyperparameters(self):

        hparams_grid = list(product(*self.hparams.values()))
        hparams_names = self.hparams.keys()
        metrics = []
        for i,params in enumerate(hparams_grid):
            logging.info(f"{i+1}/{len(hparams_grid)}")
            chparams = dict(zip(hparams_names,params))
            vmetrics = self.single_run(chparams)
            monitored_metric_val = vmetrics[self.metric]
            metrics.append(monitored_metric_val)
            self.log_metrics(i,chparams,vmetrics)
        idx_min = [i for i,o in enumerate(metrics) if o==min(metrics)]
        idx_min = idx_min[0]

        best_hparams = dict(zip(hparams_names,hparams_grid[idx_min]))
        return best_hparams, metrics
    
    def get_best_params(self):
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError("File {self.csv_path} not found please run optimization with the 'optimize_parameters'.")
         
        metrics = pd.read_csv(self.csv_path)
        idx_min = metrics[self.metric].idxmin()
        best_params = metrics.iloc[idx_min].to_dict()

        # subsetting parameters
        best_params = {k.replace("hparam/",""):v for k,v in best_params.items() if k.startswith("hparam")}
        full_params = self.fixed_args.copy()
        full_params.update(best_params)
        return full_params
       





