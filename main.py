import argparse
import torch
import os

def main():
    parser = argparse.ArgumentParser(description="2048 AI with AlphaZero approach")
    parser.add_argument('--mode', type=str, choices=['play', 'watch', 'train'], default='play')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/latest.pth')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        from training.train import train
        print("Starting training pipeline...")
        train()
    else:
        from engine.game import GameEngine
        from ui.game_ui import GameUI
        
        engine = GameEngine()
        ui = GameUI(engine)
        
        if args.mode == 'play':
            print("Running in human play mode. Use Arrow Keys to play, 'q' to quit.")
            ui.play_human()
        elif args.mode == 'watch':
            from ai.model import ZeroNet
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                
            model = ZeroNet().to(device)
            if os.path.exists(args.checkpoint):
                model.load_state_dict(torch.load(args.checkpoint, map_location=device))
                print(f"Loaded valid checkpoint: {args.checkpoint}")
            else:
                print(f"Warning: Checkpoint {args.checkpoint} not found. Proceeding with uninitialized untrained model.")
                
            model.eval()
            print("Watching the AI play using MCTS...")
            ui.play_ai(model)

if __name__ == '__main__':
    main()
