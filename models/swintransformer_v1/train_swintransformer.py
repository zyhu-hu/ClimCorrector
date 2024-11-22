import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
import modulus
from modulus.metrics.general.mse import mse
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from omegaconf import DictConfig
from modulus.launch.logging import (
    PythonLogger,
    LaunchLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from utils.data_utils import *
from climsim_datapip_processed_h5_preload import climsim_dataset_processed_h5_preload
from climsim_datapip_processed_h5 import climsim_dataset_processed_h5

# from lstm8th import LSTM8th
# import lstm8th as lstm8th

# from climsim_unet import ClimsimUnet
# import climsim_unet as climsim_unet

from swintransformer_modulus import SwinTransformerV2CrModulus
import swintransformer_modulus as swintransformer_modulus

import hydra
from torch.nn.parallel import DistributedDataParallel
from modulus.distributed import DistributedManager
from torch.utils.data.distributed import DistributedSampler
import gc
from torch.nn.utils import clip_grad_norm_
import shutil
import random

torch.set_float32_matmul_precision("high")
# Set a fixed seed value
seed = 42
# For PyTorch
torch.manual_seed(seed)
# For CUDA if using GPU
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU
# For other libraries
np.random.seed(seed)
random.seed(seed)

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:

    DistributedManager.initialize()
    dist = DistributedManager()


    data = data_utils()
    # set variables to subset
    if cfg.variable_subsets == 'v1': 
        data.set_to_v1_vars()
    elif cfg.variable_subsets == 'v2':
        data.set_to_v2_vars()
    else:
        raise ValueError('Unknown variable subset')

    # retrieve the size of the input and output
    input_size = data.input_feature_len
    output_size = data.target_feature_len
    if cfg.variable_subsets == 'v1': 
        input_size_nn = input_size + 2 # for tod and toy, I will add cos and sine of tod and toy
        output_size_nn = output_size
    elif cfg.variable_subsets == 'v2':
        input_size_nn = input_size + 3
        output_size_nn = output_size
    else:
        raise ValueError('Unknown variable subset')

    train_dataset_path = cfg.train_dataset_path
    val_dataset_path = cfg.val_dataset_path
    
    # check if train_dataset_path/**/train_input.h5 exists
    train_input_path = glob.glob(f'{train_dataset_path}/**/train_input.h5', recursive=True)
    if not train_input_path:
        raise FileNotFoundError("No 'train_input.h5' files found under the specified parent path.")
    # check if val_dataset_path/val_input.h5 exists
    val_input_path =glob.glob(f'{val_dataset_path}/**/val_input.h5')
    if not val_input_path:
        raise FileNotFoundError("No 'val_input.h5' file found under the specified parent path.")

    # val_dataset = climsim_dataset_processed_h5_preload(parent_path=val_dataset_path)
    val_dataset = climsim_dataset_processed_h5(parent_path=val_dataset_path,stage='val',target_filename=cfg.target_filename)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.distributed else None
    val_loader = DataLoader(val_dataset, 
                            batch_size=cfg.batch_size, 
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=cfg.num_workers)
    
    train_dataset = climsim_dataset_processed_h5(parent_path=train_dataset_path,stage='train',target_filename=cfg.target_filename)

    train_sampler = DistributedSampler(train_dataset) if dist.distributed else None
    
    train_loader = DataLoader(train_dataset, 
                                batch_size=cfg.batch_size, 
                                shuffle=False if dist.distributed else True,
                                sampler=train_sampler,
                                drop_last=True,
                                pin_memory=torch.cuda.is_available(),
                                num_workers=cfg.num_workers)

    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinTransformerV2CrModulus(
        img_size= (96, 144),
        patch_size = 1,
        window_size = (cfg.swin.window_size_x, cfg.swin.window_size_y),
        in_chans = cfg.swin.in_chans,
        out_chans = cfg.swin.out_chans,
        embed_dim = cfg.swin.embed_dim,
        depths = (cfg.swin.depths, ),
        num_heads = (cfg.swin.num_heads, ),
        mlp_ratio = cfg.swin.mlp_ratio,
        drop_rate = cfg.swin.drop_rate,
        full_pos_embed = False,
        rel_pos= True,
        checkpoint_stages = cfg.swin.checkpoint_stages,
        residual = False,
        random_shift = False,
    ).to(dist.device)
    # model = SwinTransformerV2CrModulus(
    #     img_size= (96, 144),
    #     patch_size = 1,
    #     window_size = (4,6),
    #     in_chans = 114,
    #     out_chans = 104,
    #     embed_dim = 256,
    #     depths = (2, ),
    #     num_heads = (8, ),
    #     mlp_ratio = 2.0,
    #     drop_rate = 0.0,
    #     full_pos_embed = False,
    #     rel_pos= True,
    #     checkpoint_stages = False,
    #     residual = False,
    #     random_shift = False,
    # ).to(dist.device)

    if len(cfg.restart_path) > 0:
        print("Restarting from checkpoint: " + cfg.restart_path)
        if dist.distributed:
            model_restart = modulus.Module.from_checkpoint(cfg.restart_path).to(dist.device)
            if dist.rank == 0:
                model.load_state_dict(model_restart.state_dict())
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                model.load_state_dict(model_restart.state_dict())
        else:
            model_restart = modulus.Module.from_checkpoint(cfg.restart_path).to(dist.device)
            model.load_state_dict(model_restart.state_dict())

    # Set up DistributedDataParallel if using more than a single process.
    # The `distributed` property of DistributedManager can be used to
    # check this.
    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],  # Set the device_id to be
                                               # the local rank of this process on
                                               # this node
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # create optimizer
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        raise ValueError('Optimizer not implemented')
    
    # create scheduler
    if cfg.scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step.step_size, gamma=cfg.scheduler.step.gamma)
    elif cfg.scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler.plateau.factor, patience=cfg.scheduler.plateau.patience, verbose=True)
    elif cfg.scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.scheduler.cosine.T_max, eta_min=cfg.scheduler.cosine.eta_min)
    elif cfg.scheduler_name == 'cosine_warmup':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.scheduler.cosine_warmup.T_0, T_mult=cfg.scheduler.cosine_warmup.T_mult, eta_min=cfg.scheduler.cosine_warmup.eta_min)
    else:
        raise ValueError('Scheduler not implemented')
    
    # create loss function
    if cfg.loss == 'mse':
        criterion = nn.MSELoss()
    elif cfg.loss == 'mae':
        criterion = nn.L1Loss()
    elif cfg.loss == 'huber':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError('Loss function not implemented')
    
    
    # Initialize the console logger
    logger = PythonLogger("main")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)

    if cfg.logger == 'wandb':
        # Initialize the MLFlow logger
        initialize_wandb(
            project=cfg.wandb.project,
            name=cfg.expname,
            entity="zeyuan_hu",
            mode="online",
        )
        LaunchLogger.initialize(use_wandb=True)

    if cfg.save_top_ckpts<=0:
        logger0.info("Checkpoints should be set >0, setting to 1")
        num_top_ckpts = 1
    else:
        num_top_ckpts = cfg.save_top_ckpts

    if cfg.top_ckpt_mode == 'min':
        top_checkpoints = [(float('inf'), None)] * num_top_ckpts
    elif cfg.top_ckpt_mode == 'max':
        top_checkpoints = [(-float('inf'), None)] * num_top_ckpts
    else:
        raise ValueError('Unknown top_ckpt_mode')
    
    if dist.rank == 0:
        save_path = os.path.join(cfg.save_path, cfg.expname) #cfg.save_path + cfg.expname
        save_path_ckpt = os.path.join(save_path, 'ckpt')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path_ckpt):
            os.makedirs(save_path_ckpt)
    
    if dist.world_size > 1:
        torch.distributed.barrier()
    
    # the following training_step and eval_step_forward are some optimized version of the training function that can take use of things like Cuda graphs, Jit, mixed precision.
    @StaticCaptureTraining(
        model=model,
        optim=optimizer,
        # cuda_graph_warmup=13,
    )
    def training_step(model, data_input, target):
        output = model(data_input)
        loss = criterion(output, target)
        return loss
    
    @StaticCaptureEvaluateNoGrad(model=model, use_graphs=False)
    def eval_step_forward(my_model, invar):
        return my_model(invar)
    
    #training block
    logger0.info("Starting Training!")
    # Basic training block with tqdm for progress tracking
    for epoch in range(cfg.epochs):
        if dist.distributed:
            train_sampler.set_epoch(epoch)

        with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=cfg.mini_batch_log_freq) as launchlog:
            model.train()

            total_iterations = len(train_loader)
            # Wrap train_loader with tqdm for a progress bar
            train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            current_step = 0
            for iteration, (data_input, target) in enumerate(train_loop):
                if cfg.early_stop_step > 0 and current_step > cfg.early_stop_step:
                    break
                data_input, target = data_input.to(device), target.to(device)
                data_input = data_input.permute(0, 3, 1, 2)
                target = target.permute(0, 3, 1, 2)

                optimizer.zero_grad()
                output = model(data_input)
                loss = criterion(output, target)
                loss.backward()

                # # for debug only, check if any parameter has None grad/not used in the backward pass
                # for name, param in model.named_parameters():
                #     if param.grad is None:
                #         print(name)

                #loss = training_step(model, data_input, target)

                if cfg.clip_grad:
                    clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_norm)

                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
                optimizer.step()
                if cfg.scheduler_name == 'cosine_warmup':
                    scheduler.step(epoch + iteration / total_iterations)

                launchlog.log_minibatch({"loss_train": loss.detach().cpu().numpy(), "lr": optimizer.param_groups[0]["lr"], "total_norm": total_norm.item()})

                # Update the progress bar description with the current loss
                train_loop.set_description(f'Epoch {epoch+1}')
                train_loop.set_postfix(loss=loss.item())
                current_step += 1
            
            model.eval()
            val_loss = 0.0
            num_samples_processed = 0
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/1 [Validation]')
            current_step = 0
            for data_input, target in val_loop:
                if cfg.early_stop_step > 0 and current_step > cfg.early_stop_step:
                    break
                data_input, target = data_input.to(device), target.to(device)
                data_input = data_input.permute(0, 3, 1, 2)
                target = target.permute(0, 3, 1, 2)
                output = eval_step_forward(model, data_input)
                loss = criterion(output, target)
                val_loss += loss.item() * data_input.size(0)
                num_samples_processed += data_input.size(0)

                # Calculate and update the current average loss
                current_val_loss_avg = val_loss / num_samples_processed
                val_loop.set_postfix(loss=current_val_loss_avg)
                current_step += 1
                # del data_input, target, output
                    
            
            # if dist.rank == 0:
                #all reduce the loss
            if dist.world_size > 1:
                current_val_loss_avg = torch.tensor(current_val_loss_avg, device=dist.device)
                torch.distributed.all_reduce(current_val_loss_avg)
                current_val_loss_avg = current_val_loss_avg.item() / dist.world_size

            if dist.rank == 0:
                launchlog.log_epoch({"loss_valid": current_val_loss_avg})

                current_metric = current_val_loss_avg
                # Save the top checkpoints
                if cfg.top_ckpt_mode == 'min':
                    is_better = current_metric < max(top_checkpoints, key=lambda x: x[0])[0]
                elif cfg.top_ckpt_mode == 'max':
                    is_better = current_metric > min(top_checkpoints, key=lambda x: x[0])[0]
                
                if len(top_checkpoints) == 0 or is_better:
                    ckpt_path = os.path.join(save_path_ckpt, f'ckpt_epoch_{epoch+1}_metric_{current_metric:.4f}.mdlus')
                    if dist.distributed:
                        model.module.save(ckpt_path)
                    else:
                        model.save(ckpt_path)
                    # if dist.rank == 0:
                    #     model.save(ckpt_path)
                    top_checkpoints.append((current_metric, ckpt_path))
                    # Sort and keep top 5 based on max/min goal at the beginning
                    if cfg.top_ckpt_mode == 'min':
                        top_checkpoints.sort(key=lambda x: x[0], reverse=False)
                    elif cfg.top_ckpt_mode == 'max':
                        top_checkpoints.sort(key=lambda x: x[0], reverse=True)
                    # delete the worst checkpoint
                    if len(top_checkpoints) > num_top_ckpts:
                        worst_ckpt = top_checkpoints.pop()
                        print(f"Removing worst checkpoint: {worst_ckpt[1]}")
                        if worst_ckpt[1] is not None:
                            os.remove(worst_ckpt[1])
                            
            if cfg.scheduler_name == 'plateau':
                scheduler.step(current_val_loss_avg)
            elif cfg.scheduler_name == 'cosine_warmup':
                pass # handled in the optimizer.step() in training loop
            else:
                scheduler.step()
            
            if dist.world_size > 1:
                torch.distributed.barrier()
                
    if dist.rank == 0:
        logger0.info("Start recovering the model from the top checkpoint to do torchscript conversion")         
        #recover the model weight to the top checkpoint
        model = modulus.Module.from_checkpoint(top_checkpoints[0][1]).to(device)

        # Save the model
        save_file = os.path.join(save_path, 'model.mdlus')
        model.save(save_file)
        # convert the model to torchscript
        swintransformer_modulus.device = "cpu"
        device = torch.device("cpu")
        model_inf = modulus.Module.from_checkpoint(save_file).to(device)
        scripted_model = torch.jit.script(model_inf)
        scripted_model = scripted_model.eval()
        save_file_torch = os.path.join(save_path, 'model.pt')
        scripted_model.save(save_file_torch)
        # # copy input and output normalizations files
        # shutil.copy(cfg.input_mean, save_path+'/input_mean.npy')
        # shutil.copy(cfg.input_std, save_path+'/input_std.npy')
        # shutil.copy(cfg.target_mean, save_path+'/target_mean.npy')
        # shutil.copy(cfg.target_std, save_path+'/target_std.npy')
        
        logger0.info("saved model to: " + save_path)

        logger0.info("Training complete!")

    return current_val_loss_avg

if __name__ == "__main__":
    main()