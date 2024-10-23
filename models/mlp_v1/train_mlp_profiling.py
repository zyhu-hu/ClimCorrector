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
from climsim_datapip_h5_preload import climsim_dataset_h5_preload
from climsim_datapip_h5 import climsim_dataset_h5

from mlp import MLP
import mlp as mlp
import hydra
from torch.nn.parallel import DistributedDataParallel
from modulus.distributed import DistributedManager
from torch.utils.data.distributed import DistributedSampler
import gc
from torch.nn.utils import clip_grad_norm_
import shutil
import nvtx

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:

    DistributedManager.initialize()
    dist = DistributedManager()


    data = data_utils()
    # set variables to subset
    if cfg.variable_subsets == 'v1': 
        data.set_to_v1_vars()
    else:
        raise ValueError('Unknown variable subset')

    # retrieve the size of the input and output
    input_size = data.input_feature_len
    output_size = data.target_feature_len
    if cfg.variable_subsets == 'v1': 
        input_size_nn = input_size + 2 # for tod and toy, I will add cos and sine of tod and toy
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
    val_input_path = f'{val_dataset_path}/val_input.h5'
    if not os.path.exists(val_input_path):
        raise FileNotFoundError("No 'val_input.h5' file found under the specified parent path.")
    
    # read the normalization values
    input_mean_array = np.load(cfg.input_mean)
    input_std_array = np.load(cfg.input_std)
    target_mean_array = np.load(cfg.target_mean)
    target_std_array = np.load(cfg.target_std)

    # check if the normalization array sizes match the input/output sizes
    if input_mean_array.shape[0] != input_size or input_std_array.shape[0] != input_size or target_mean_array.shape[0] != output_size or target_std_array.shape[0] != output_size:
        raise ValueError('Normalization array sizes do not match the input/output sizes')

    val_dataset = climsim_dataset_h5_preload(parent_path=val_dataset_path, 
                 input_mean=input_mean_array,
                 input_std=input_std_array,
                 target_mean=target_mean_array,
                 target_std=target_std_array,         
                 input_clip=cfg.input_clip,
                 target_clip=cfg.target_clip,
                 target_clip_value=cfg.target_clip_value)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.distributed else None
    val_loader = DataLoader(val_dataset, 
                            batch_size=cfg.batch_size, 
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=cfg.num_workers)
    
    train_dataset = climsim_dataset_h5(parent_path=train_dataset_path, 
                 input_mean=input_mean_array,
                 input_std=input_std_array,
                 target_mean=target_mean_array,
                 target_std=target_std_array,         
                 input_clip=cfg.input_clip,
                 target_clip=cfg.target_clip,
                 target_clip_value=cfg.target_clip_value)

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

    model = MLP(
        in_dims=input_size_nn,
        out_dims=output_size_nn,
        hidden_dims=cfg.mlp.hidden_dims,
        layers=cfg.mlp.layers,
    ).to(device)

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
        # cuda_graph_warmup=11,
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
        
        # NVTX annotation for the entire training phase
        torch.cuda.nvtx.range_push(f"Epoch {epoch+1} Training")

        with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=cfg.mini_batch_log_freq) as launchlog:
            model.train()

            total_iterations = len(train_loader)
            # Wrap train_loader with tqdm for a progress bar
            # train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            current_step = 0
            # Create an iterator for train_loader
            train_loader_iter = iter(train_loader)
            # for iteration, (data_input, target) in enumerate(train_loop):
            for iteration in range(len(train_loader)):
                # Use NVTX range for data loading

                torch.cuda.nvtx.range_push(f"Data Loading Iteration {iteration+1}")
                data_input, target = next(train_loader_iter)  # Get the next batch
                torch.cuda.nvtx.range_pop() # End of Data Loading

                if cfg.early_stop_step > 0 and current_step > cfg.early_stop_step:
                    break

                # NVTX annotation for each batch processing in training
                torch.cuda.nvtx.range_push(f"Training Step {current_step+1}")

                data_input, target = data_input.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data_input)
                loss = criterion(output, target)
                loss.backward()
                #loss = training_step(model, data_input, target)

                if cfg.clip_grad:
                    clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_norm)

                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
                optimizer.step()
                if cfg.scheduler_name == 'cosine_warmup':
                    scheduler.step(epoch + iteration / total_iterations)

                launchlog.log_minibatch({"loss_train": loss.detach().cpu().numpy(), "lr": optimizer.param_groups[0]["lr"], "total_norm": total_norm.item()})

                # Update the progress bar description with the current loss
                # train_loop.set_description(f'Epoch {epoch+1}')
                # train_loop.set_postfix(loss=loss.item())
                current_step += 1

                torch.cuda.nvtx.range_pop()  # End of Training Step
            
            # NVTX annotation for the evaluation phase
            torch.cuda.nvtx.range_push(f"Epoch {epoch+1} Validation")
            model.eval()
            val_loss = 0.0
            num_samples_processed = 0
            # val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/1 [Validation]')
            # current_step = 0
            # for data_input, target in val_loop:
            #     if cfg.early_stop_step > 0 and current_step > cfg.early_stop_step:
            #         break
            #     data_input, target = data_input.to(device), target.to(device)

            val_loader_iter = iter(val_loader)
            # for iteration, (data_input, target) in enumerate(train_loop):
            for iteration in range(len(val_loader)):
                # Use NVTX range for data loading

                torch.cuda.nvtx.range_push(f"Data Loading val Iteration {iteration+1}")
                data_input, target = next(val_loader_iter)  # Get the next batch
                torch.cuda.nvtx.range_pop() # End of Data Loading

                if cfg.early_stop_step > 0 and current_step > cfg.early_stop_step:
                     break
                
                torch.cuda.nvtx.range_push(f"Validation Step {current_step+1}")

                data_input, target = data_input.to(device), target.to(device)

                output = eval_step_forward(model, data_input)
                loss = criterion(output, target)
                val_loss += loss.item() * data_input.size(0)
                num_samples_processed += data_input.size(0)

                # Calculate and update the current average loss
                current_val_loss_avg = val_loss / num_samples_processed
                # val_loop.set_postfix(loss=current_val_loss_avg)
                current_step += 1
                torch.cuda.nvtx.range_pop()
                # del data_input, target, output

            torch.cuda.nvtx.range_pop()  # End of Validation phase
                    
            
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

        torch.cuda.nvtx.range_pop()  # End of Training Epoch

    if dist.rank == 0:
        logger0.info("Start recovering the model from the top checkpoint to do torchscript conversion")         
        #recover the model weight to the top checkpoint
        model = modulus.Module.from_checkpoint(top_checkpoints[0][1]).to(device)

        # Save the model
        save_file = os.path.join(save_path, 'model.mdlus')
        model.save(save_file)
        # convert the model to torchscript
        mlp.device = "cpu"
        device = torch.device("cpu")
        model_inf = modulus.Module.from_checkpoint(save_file).to(device)
        scripted_model = torch.jit.script(model_inf)
        scripted_model = scripted_model.eval()
        save_file_torch = os.path.join(save_path, 'model.pt')
        scripted_model.save(save_file_torch)
        # copy input and output normalizations files
        shutil.copy(cfg.input_mean, save_path+'/input_mean.npy')
        shutil.copy(cfg.input_std, save_path+'/input_std.npy')
        shutil.copy(cfg.target_mean, save_path+'/target_mean.npy')
        shutil.copy(cfg.target_std, save_path+'/target_std.npy')
        
        logger0.info("saved input/output normalizations and model to: " + save_path)

        logger0.info("Training complete!")

    return current_val_loss_avg

if __name__ == "__main__":
    main()