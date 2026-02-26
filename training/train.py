import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from ai.model import ZeroNet
from training.replay_buffer import ReplayBuffer
from training.self_play import play_game
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def worker_play_games(worker_args):
    state_dict, mcts_sims, num_games, worker_id = worker_args
    # Instantiate model on CPU for multiprocessing stability
    model = ZeroNet()
    model.load_state_dict(state_dict)
    model.eval()
    
    all_states, all_policies, all_values = [], [], []
    for g in range(num_games):
        # We append worker_id for better logging
        s, p, v = play_game(model, num_simulations=mcts_sims, game_idx=g+1, total_games=num_games)
        all_states.extend(s)
        all_policies.extend(p)
        all_values.extend(v)
        
    return all_states, all_policies, all_values

def train(iterations=50, games_per_iteration=50, epochs=5, batch_size=64, mcts_sims=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    model = ZeroNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=20000)
    
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = 'checkpoints/latest.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded checkpoint.")

    print(f"Training on {device}")
    
    for i in range(iterations):
        print(f"--- Iteration {i+1}/{iterations} ---")
        model.eval()
        
        # Self-play phase
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
        
        games_per_worker = [games_per_iteration // num_cores] * num_cores
        for j in range(games_per_iteration % num_cores):
            games_per_worker[j] += 1
            
        worker_args_list = []
        for j, n_games in enumerate(games_per_worker):
            if n_games > 0:
                worker_args_list.append((state_dict, mcts_sims, n_games, j))
                
        print(f"Starting {games_per_iteration} games distributed across {len(worker_args_list)} workers...")
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(worker_play_games, worker_args_list))
            
        for s_list, p_list, v_list in results:
            for s, p, v in zip(s_list, p_list, v_list):
                buffer.push(s, p, v)
                
        print(f"==> Buffer size: {len(buffer)}")
                
        # Train phase
        model.train()
        if len(buffer) < batch_size:
            print("Not enough data to train. Skipping training phase.")
            continue
            
        total_loss = 0
        for epoch in range(epochs):
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
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / epochs
        print(f"Training Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), checkpoint_path)
        print("Saved checkpoint.\n")

if __name__ == '__main__':
    train()
