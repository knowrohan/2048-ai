"""
Main training loop for the AI model.
Coordinates self-play data generation and neural network weight updates
using experience replay, optimizing both policy and value losses.
References `ai.model.ZeroNet`, `training.replay_buffer`, and `training.self_play`.
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from ai.model import ZeroNet
from training.replay_buffer import ReplayBuffer
from training.self_play import play_games_concurrently

def train(iterations=20, games_per_iteration=10, epochs=5, batch_size=128, mcts_sims=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    model = ZeroNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Cosine annealing learning rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=1e-5)
    
    buffer = ReplayBuffer(capacity=100000)
    
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = 'checkpoints/latest.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_iter = checkpoint.get('iteration', 0)
            print(f"Resumed from checkpoint at iteration {start_iter}.")
        else:
            # Legacy checkpoint: just model state_dict
            model.load_state_dict(checkpoint)
            start_iter = 0
            print("Loaded legacy checkpoint (model weights only).")
    else:
        start_iter = 0

    print(f"Training on {device}")
    print(f"Config: {iterations} iters, {games_per_iteration} games/iter, {mcts_sims} MCTS sims, {epochs} epochs, batch_size={batch_size}")
    
    # Max concurrent games limited by games_per_iteration, or a fixed constant (e.g. 256)
    # This prevents creating 256 game buffers if we only requested 50 games
    num_concurrent_games = min(256, games_per_iteration)
    
    for i in range(start_iter, iterations):
        print(f"\n--- Iteration {i+1}/{iterations} (lr={scheduler.get_last_lr()[0]:.6f}) ---")
        model.eval()
        
        print(f"Starting {games_per_iteration} games via Batched GPU Self-Play (Max concurrent: {num_concurrent_games})...")
        
        # Single process, GPU batched play
        states, policies, values = play_games_concurrently(
            model=model, 
            target_games=games_per_iteration,
            num_concurrent=num_concurrent_games,
            num_simulations=mcts_sims
        )
            
        for s, p, v in zip(states, policies, values):
            buffer.push(s, p, v)
                
        print(f"==> Buffer size: {len(buffer)}")
                
        # Train phase
        model.train()
        if len(buffer) < batch_size:
            print("Not enough data to train. Skipping training phase.")
            continue
            
        total_loss = 0
        num_batches = max(1, len(buffer) // batch_size)
        num_batches = min(num_batches, 300)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(num_batches):
                states, policies, values = buffer.sample(batch_size)
                states = states.to(device)
                policies = policies.to(device)
                values = values.to(device)
                
                pred_policies, pred_values = model(states)
                
                policy_loss = -torch.sum(policies * F.log_softmax(pred_policies, dim=1), dim=1).mean()
                value_loss = F.mse_loss(pred_values, values)
                loss = policy_loss + value_loss
                
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            total_loss += epoch_loss / num_batches
            
        avg_loss = total_loss / epochs
        print(f"Training Loss: {avg_loss:.4f}")
        
        # Step LR scheduler after each iteration
        scheduler.step()
        
        # Save full checkpoint with optimizer and scheduler state
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'iteration': i + 1,
        }, checkpoint_path)
        print("Saved checkpoint.")

if __name__ == '__main__':
    train()