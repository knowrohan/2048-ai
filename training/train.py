"""
Main training loop for the AI model.
Coordinates self-play data generation and neural network weight updates
using experience replay, optimizing both policy and value losses.
References `ai.model.ZeroNet`, `training.replay_buffer`, and `training.self_play`.
"""
import os
import argparse
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from ai.model import ZeroNet
from training.replay_buffer import ReplayBuffer
from training.self_play import play_games_concurrently

def train(iterations=20, games_per_iteration=20, epochs=10, batch_size=256,
          mcts_sims_start=15, mcts_sims_end=100, num_concurrent=64, augment_factor=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    model = ZeroNet().to(device)
    
    # Load checkpoint BEFORE torch.compile to avoid state_dict key mismatch
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = 'checkpoints/latest.pth'
    start_iter = 0
    loaded_optimizer_state = None
    loaded_scheduler_state = None
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            loaded_optimizer_state = checkpoint['optimizer_state_dict']
            loaded_scheduler_state = checkpoint['scheduler_state_dict']
            start_iter = checkpoint.get('iteration', 0)
            print(f"Resumed from checkpoint at iteration {start_iter}.")
        else:
            # Legacy checkpoint: just model state_dict
            model.load_state_dict(checkpoint)
            print("Loaded legacy checkpoint (model weights only).")

    # JIT-compile model for faster inference (PyTorch 2.0+, CUDA only — MPS not supported)
    if device.type == 'cuda':
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        except Exception:
            print("torch.compile failed, using eager mode")
    else:
        print(f"Skipping torch.compile (not supported on {device.type})")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=1e-5)
    
    if loaded_optimizer_state:
        optimizer.load_state_dict(loaded_optimizer_state)
    if loaded_scheduler_state:
        scheduler.load_state_dict(loaded_scheduler_state)
    
    buffer = ReplayBuffer(capacity=100000)

    print(f"Training on {device}")
    print(f"Config: {iterations} iters, {games_per_iteration} games/iter, "
          f"MCTS sims {mcts_sims_start}→{mcts_sims_end}, {epochs} epochs, "
          f"batch_size={batch_size}, concurrent={num_concurrent}, augment={augment_factor}x")
    
    for i in range(start_iter, iterations):
        # Progressive MCTS simulations: linearly ramp from start to end
        if iterations > 1:
            progress = i / (iterations - 1)
        else:
            progress = 1.0
        current_sims = int(mcts_sims_start + progress * (mcts_sims_end - mcts_sims_start))
        
        iter_start = time.time()
        print(f"\n{'='*80}")
        print(f"  Iteration {i+1}/{iterations} | lr={scheduler.get_last_lr()[0]:.6f} | sims={current_sims}")
        print(f"{'='*80}")
        model.eval()
        
        effective_concurrent = min(num_concurrent, games_per_iteration)
        print(f"\n[Self-Play] {games_per_iteration} games ({effective_concurrent} concurrent, {current_sims} MCTS sims)")
        
        # Single-process concurrent self-play (no multiprocessing overhead)
        selfplay_start = time.time()
        states, policies, values = play_games_concurrently(
            model=model, 
            target_games=games_per_iteration,
            num_concurrent=effective_concurrent,
            num_simulations=current_sims,
            augment_factor=augment_factor
        )
        selfplay_time = time.time() - selfplay_start
            
        for s, p, v in zip(states, policies, values):
            buffer.push(s, p, v)
                
        print(f"  Buffer size: {len(buffer)} samples")
                
        # Train phase
        model.train()
        if len(buffer) < batch_size:
            print("Not enough data to train. Skipping training phase.")
            continue
            
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = max(1, len(buffer) // batch_size)
        num_batches = min(num_batches, 300)
        
        print(f"\n[Training] {epochs} epochs × {num_batches} batches (batch_size={batch_size})")
        train_start = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0
            epoch_policy_loss = 0
            epoch_value_loss = 0
            
            for _ in range(num_batches):
                states, policies, values = buffer.sample(batch_size)
                states = states.to(device, non_blocking=True)
                policies = policies.to(device, non_blocking=True)
                values = values.to(device, non_blocking=True)
                
                pred_policies, pred_values = model(states)
                
                policy_loss = -torch.sum(policies * F.log_softmax(pred_policies, dim=1), dim=1).mean()
                value_loss = F.mse_loss(pred_values, values)
                loss = policy_loss + value_loss
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_ploss = epoch_policy_loss / num_batches
            avg_epoch_vloss = epoch_value_loss / num_batches
            epoch_time = time.time() - epoch_start
            
            total_loss += avg_epoch_loss
            total_policy_loss += avg_epoch_ploss
            total_value_loss += avg_epoch_vloss
            
            print(f"  Epoch {epoch+1}/{epochs} │ Loss: {avg_epoch_loss:.4f} (policy: {avg_epoch_ploss:.4f}, value: {avg_epoch_vloss:.4f}) │ {epoch_time:.1f}s")
            
        train_time = time.time() - train_start
        avg_loss = total_loss / epochs
        avg_ploss = total_policy_loss / epochs
        avg_vloss = total_value_loss / epochs
        print(f"  Training complete: avg loss {avg_loss:.4f} (policy: {avg_ploss:.4f}, value: {avg_vloss:.4f}) in {train_time:.1f}s")
        
        # Step LR scheduler after each iteration
        scheduler.step()
        
        # Save full checkpoint with optimizer and scheduler state
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'iteration': i + 1,
        }, checkpoint_path)
        
        iter_time = time.time() - iter_start
        print(f"\n  Iteration {i+1} complete in {iter_time:.1f}s (self-play: {selfplay_time:.1f}s, training: {train_time:.1f}s)")
        print(f"  Checkpoint saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train 2048 AI")
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--games_per_iteration', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mcts_sims_start', type=int, default=15, help="MCTS sims at iteration 0")
    parser.add_argument('--mcts_sims_end', type=int, default=100, help="MCTS sims at final iteration")
    parser.add_argument('--mcts_sims', type=int, default=None, help="Fixed MCTS sims (overrides start/end)")
    parser.add_argument('--num_concurrent', type=int, default=64, help="Concurrent games in self-play")
    parser.add_argument('--augment_factor', type=int, default=8, choices=[1, 2, 8], help="Data augmentation factor")
    args = parser.parse_args()

    sims_start = args.mcts_sims_start
    sims_end = args.mcts_sims_end
    if args.mcts_sims is not None:
        sims_start = args.mcts_sims
        sims_end = args.mcts_sims

    train(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mcts_sims_start=sims_start,
        mcts_sims_end=sims_end,
        num_concurrent=args.num_concurrent,
        augment_factor=args.augment_factor,
    )