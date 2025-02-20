import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import modulus
from modulus.metrics.general.mse import mse
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from omegaconf import DictConfig
from modulus.launch.logging import (
    PythonLogger,
    LaunchLogger,
    initialize_wandb,
)
from utils.data_utils import *
from climsim_datapip_processed_h5 import climsim_dataset_processed_h5

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
from soap import SOAP
from modulus.launch.utils import load_checkpoint, save_checkpoint

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

    train_dataset_path = cfg.train_dataset_path
    val_dataset_path = cfg.val_dataset_path
    
    # check if train_dataset_path/**/train_input.h5 exists
    train_input_path = glob.glob(f'{train_dataset_path}/**/train_input.h5', recursive=True)
    if not train_input_path:
        raise FileNotFoundError("No 'train_input.h5' files found under the specified parent path.")
    # check if val_dataset_path/val_input.h5 exists
    # note that here I assumed there is one or a few subfolders under val_dataset_path that contains val_input.h5
    val_input_path =glob.glob(f'{val_dataset_path}/**/val_input.h5')
    if not val_input_path:
        raise FileNotFoundError("No 'val_input.h5' file found under the specified parent path.")

    # create the distributed validation dataloader below
    val_dataset = climsim_dataset_processed_h5(parent_path=val_dataset_path,stage='val',target_filename=cfg.target_filename)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.distributed else None
    val_loader = DataLoader(val_dataset, 
                            batch_size=cfg.batch_size, 
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=cfg.num_workers)
    
    # create the distributed training dataloader below
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

    # create optimizer
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'soap':
        optimizer = SOAP(model.parameters(), lr = cfg.learning_rate, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
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

    # create paths to save model weights
    # save_path will save the final model weights. In my code, this final saved model will be the one that has the best validation score
    # save_path_ckpt will save model weights after every epoch
    # save_path_ckpt_full will save the entire training state after every epoch: including model weights, optimizer and scheduler state. 
    # Such full training state and allow the training to restart after any break without the need to warmup the optimizer

    save_path = os.path.join(cfg.save_path, cfg.expname) #cfg.save_path + cfg.expname
    save_path_ckpt = os.path.join(save_path, 'ckpt')
    save_path_ckpt_full = os.path.join(save_path_ckpt, 'ckpt_full')
    if dist.rank == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path_ckpt):
            os.makedirs(save_path_ckpt)
        if not os.path.exists(save_path_ckpt_full):
            os.makedirs(save_path_ckpt_full)

    # if cfg.restart_path is set, then the model will load the model weight in the restart_path
    # Note that in this restart method, the optimizer is not warmed up (e.g., for adam optimizer, you typically need some training steps for the adam optimizer to start perform well)
    # below there is also another restart method by setting cfg.restart_full_ckpt, which will load both model weights and optimizer/scheduler state if you saved such checkpoint
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

    # if cfg.restart_full_ckpt is set, then the code below will load the model weights as well as the optimizer/scheduler state
    # you won't need to set cfg.restart_path if you set the cfg.restart_full_ckpt
    if cfg.restart_full_ckpt:
        print("Restarting from full checkpoint: " + save_path_ckpt_full)
        loaded_epoch = load_checkpoint(
            save_path_ckpt_full,
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cuda",
        )
    else:
        loaded_epoch = 0

    
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
    if cfg.logger == 'wandb':
        # Initialize the MLFlow logger
        initialize_wandb(
            project=cfg.wandb.project,
            name=cfg.expname,
            entity="zeyuan_hu",
            mode="online",
        )
        LaunchLogger.initialize(use_wandb=True)



    # set how many top checkpoints to track
    if cfg.save_top_ckpts<=0:
        logger.info("Checkpoints should be set >0, setting to 1")
        num_top_ckpts = 1
    else:
        num_top_ckpts = cfg.save_top_ckpts
    if cfg.top_ckpt_mode == 'min':
        top_checkpoints = [(float('inf'), None)] * num_top_ckpts
    elif cfg.top_ckpt_mode == 'max':
        top_checkpoints = [(-float('inf'), None)] * num_top_ckpts
    else:
        raise ValueError('Unknown top_ckpt_mode')
    
    if dist.world_size > 1:
        torch.distributed.barrier()
    
    # # the following training_step and eval_step_forward are some optimized version of the training function that can take use of things like Cuda graphs, Jit, mixed precision.
    # they are provided by the modulus library. However, there are some issue using them with the swintransformer which I have not debugged successfully. So I commented out their usage here.
    # there usage can be found in this example: https://docs.nvidia.com/deeplearning/modulus/modulus-core/tutorials/simple_training_example.html#optimized-training-workflow

    # @StaticCaptureTraining(
    #     model=model,
    #     optim=optimizer,
    #     # cuda_graph_warmup=13,
    # )
    # def training_step(model, data_input, target):
    #     output = model(data_input)
    #     loss = criterion(output, target)
    #     return loss
    
    # @StaticCaptureEvaluateNoGrad(model=model, use_graphs=False)
    # def eval_step_forward(my_model, invar):
    #     return my_model(invar)
    
    #training block
    logger.info("Starting Training!")
    # Basic training block with tqdm for progress tracking
    for epoch in range(max(1,loaded_epoch+1),cfg.epochs+1):
        if dist.distributed:
            train_sampler.set_epoch(epoch)

        with LaunchLogger("train", epoch=epoch, mini_batch_log_freq=cfg.mini_batch_log_freq) as launchlog:
            model.train()

            total_iterations = len(train_loader)
            # Wrap train_loader with tqdm for a progress bar
            train_loop = tqdm(train_loader, desc=f'Epoch {epoch}')
            current_step = 0
            for iteration, (data_input, target) in enumerate(train_loop):
                # here I added an early stop option. if you set cfg.early_stop_step>0 (default is -1), then each training epoch will only have cfg.early_stop_step steps. 
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

                # below is an option to do gradient clipping, which can be useful if you found very large gradient at some steps (e.g., when there is a outlier in data) that can potentially disrupt the training.
                if cfg.clip_grad:
                    clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad_norm)

                optimizer.step()
                if cfg.scheduler_name == 'cosine_warmup':
                    # here in cosine_warmup scheduler, I wanted to let the learning rate to change every training steps instead of every epoch. Just an arbitrary choice.
                    scheduler.step(epoch + iteration / total_iterations)

                launchlog.log_minibatch({"loss_train": loss.detach().cpu().numpy(), "lr": optimizer.param_groups[0]["lr"]})

                # Update the progress bar description with the current loss
                train_loop.set_description(f'Epoch {epoch}')
                train_loop.set_postfix(loss=loss.item())
                current_step += 1
            
            model.eval()
            val_loss = 0.0
            num_samples_processed = 0
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch}/1 [Validation]')
            for data_input, target in val_loop:
                data_input, target = data_input.to(device), target.to(device)
                data_input = data_input.permute(0, 3, 1, 2)
                target = target.permute(0, 3, 1, 2)
                output = model(data_input)
                loss = criterion(output, target)
                val_loss += loss.item() * data_input.size(0)
                num_samples_processed += data_input.size(0)

                # Calculate and update the current average loss
                current_val_loss_avg = val_loss / num_samples_processed
                val_loop.set_postfix(loss=current_val_loss_avg)

            
            # for validation, we need to calculate the validation score over the entire validation dataset.
            # since each gpu only usage a fraction of the validation set, we need to gather the validation score on each gpu and do an average using the all_reduce method below
            if dist.world_size > 1:
                current_val_loss_avg = torch.tensor(current_val_loss_avg, device=dist.device)
                torch.distributed.all_reduce(current_val_loss_avg)
                current_val_loss_avg = current_val_loss_avg.item() / dist.world_size

            if dist.rank == 0:
                launchlog.log_epoch({"loss_valid": current_val_loss_avg})

                current_metric = current_val_loss_avg
                # Save the top checkpoints
                # this part of the code is a bit arbitrary. I choose to save the top n=cfg.save_top_ckpts checkpoints based on their validation score.
                if cfg.top_ckpt_mode == 'min':
                    is_better = current_metric < max(top_checkpoints, key=lambda x: x[0])[0]
                elif cfg.top_ckpt_mode == 'max':
                    is_better = current_metric > min(top_checkpoints, key=lambda x: x[0])[0]
                
                if len(top_checkpoints) == 0 or is_better:
                    ckpt_path = os.path.join(save_path_ckpt, f'ckpt_epoch_{epoch}_metric_{current_metric:.4f}.mdlus')
                    if dist.distributed:
                        model.module.save(ckpt_path)
                    else:
                        model.save(ckpt_path)

                    top_checkpoints.append((current_metric, ckpt_path))
                    # Sort and keep top n=cfg.save_top_ckpts checkpoints
                    if cfg.top_ckpt_mode == 'min':
                        top_checkpoints.sort(key=lambda x: x[0], reverse=False)
                    elif cfg.top_ckpt_mode == 'max':
                        top_checkpoints.sort(key=lambda x: x[0], reverse=True)
                    # delete the worst checkpoint if the saved checkpoints exceed cfg.save_top_ckpts
                    if len(top_checkpoints) > num_top_ckpts:
                        worst_ckpt = top_checkpoints.pop()
                        print(f"Removing worst checkpoint: {worst_ckpt[1]}")
                        if worst_ckpt[1] is not None:
                            os.remove(worst_ckpt[1])
                            
            if cfg.scheduler_name == 'plateau':
                # note that the reduceonplateau learning rate scheduler needs to adjust the learning rate based on if the validation score has not been improved over certain epochs.
                scheduler.step(current_val_loss_avg)
            elif cfg.scheduler_name == 'cosine_warmup':
                pass # handled in the optimizer.step() in training loop
            else:
                scheduler.step()
            
            if dist.world_size > 1:
                torch.distributed.barrier()

            # add saving full checkpoint including optimizer/scheduler state
            if cfg.restart_full_ckpt:
                if dist.rank == 0:
                    save_checkpoint(
                        save_path_ckpt_full,
                        models=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                    )
                
    if dist.rank == 0:
        logger.info("Start recovering the model from the top checkpoint")         
        #recover the model weight to the top checkpoint
        model = modulus.Module.from_checkpoint(top_checkpoints[0][1]).to(device)

        # Save the model at save_path
        save_file = os.path.join(save_path, 'model.mdlus')
        model.save(save_file)

        logger.info("saved model to: " + save_path)
        logger.info("Training complete!")

    return current_val_loss_avg

if __name__ == "__main__":
    main()