import os
from glob import glob
import importer
from pathlib import Path
import argparse
import hydra
from omegaconf import OmegaConf
import torch
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from src.models import LitModel
from deq.standard.lib.solvers import anderson, forward_iteration
import pandas as pd
import logging

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu") 

log = logging.getLogger(__name__)

# def parse_args():
#     parser = argparse.ArgumentParser(description='Plot 2D loss surface')
#     parser.add_argument('--surf_file', '-f', default='', help='The h5 file that contains surface values')
#     parser.add_argument('--dir_file', default='', help='The h5 file that contains directions')
#     parser.add_argument('--proj_file', default='', help='The h5 file that contains the projected trajectories')
#     parser.add_argument('--surf_name', default='train_loss', help='The type of surface to plot')
#     parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
#     parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
#     parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
#     parser.add_argument('--zlim', default=10, type=float, help='Maximum loss value to show')
#     parser.add_argument('--show', action='store_true', default=False, help='show plots')

#     args = parser.parse_args()

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
    ckpts = glob("/home/user/code/TheDynamic/logs/experiments/fractal_fp2_*tanh*/checkpoints/last.ckpt", recursive=True)
    print(ckpts)

    output = Path('logs/eval.csv')
    new_csv = False
    if not output.exists():
        new_csv = True
        output.parent.mkdir(exist_ok=True)
        df = pd.DataFrame(columns=('ckpt', ))
    else:
        df = pd.read_csv(output)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    for ckpt in ckpts:
        if ckpt in df['ckpt']: continue
        ckpt_dir = Path(os.path.dirname(ckpt))
        exp_dir = ckpt_dir.parent.absolute()
        config_file = exp_dir.joinpath('.hydra/config.yaml')
        config = OmegaConf.load(config_file)
        config.checkpoint = ckpt
        df_row = {'ckpt': ckpt}

        # device = torch.device("cpu")
        # Init lightning datamodule
        datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
        datamodule.setup()
        test_dl = datamodule.test_dataloader()
        df_row['data_dim'] = datamodule.hparams.dim
        df_row['data_rho'] = datamodule.hparams.rho


        log.info(f"Loading weights from checkpoint <{config.checkpoint}>")
        
        model = LitModel.load_from_checkpoint(config.checkpoint)
        # model.model.core.save_result=True
        df_row['h_features'] = config.model.arch.h_features
        df_row['init_std'] = config.model.init_std
        df_row['train_f_solver'] = model.model.core.f_solver.__name__
        df_row['train_b_solver'] = model.model.core.b_solver.__name__
        
        # Init lightning trainer
        log.info(f"Instantiating trainer <{config.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, weights_summary=None, logger=None, callbacks=None, _convert_="partial", progress_bar_refresh_rate=0
        )
        
        for f_solver in ['anderson', 'forward_iteration']:
            model.model.core.f_solver = eval(f_solver)
            scores = trainer.test(model, test_dl, verbose=False)[0]
            for k, v in scores.items():
                df_row[f'{f_solver}|{k}'] = v
        pd.DataFrame().append(df_row, ignore_index=True).to_csv(output, mode='a', header=new_csv, index=False)
        new_csv = False

if __name__ == '__main__':
    main()