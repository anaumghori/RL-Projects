import os
from src.trainer import Trainer

def main():
    trainer = Trainer()
    print("Training for 500 episodes...")
    trainer.train(episodes=500)
    trainer.save_model('results/model.pth')
    print("Model saved!")
    print("\nTesting the trained agent...")
    trainer.test_agent(episodes=5)

if __name__ == "__main__":
    main()