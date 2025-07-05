# Cat Quest  
Implementation of simple **Deep Q-Network** and a custom environment where a cat must navigate a grid to reach meat while avoiding obstacles like bathtubs and brooms. Easily adjust difficulty by modifying hyperparameters in [configs/config.py](https://github.com/anaumghori/RL-Projects/blob/main/CatQuest/configs/config.py)    
#### Initial Grid Layout
🐱 ⬜ ⬜   
⬜ ⬜ 🧹   
🛁 ⬜ ⬜   
⬜ 🥩 ⬜  
### File Structure
```
├── src/  
│   ├── environment.py           # Complete puzzle environment  
│   ├── dqn_agent.py             # DQN network + agent logic  
│   ├── trainer.py               # Training loop and evaluation  
├── configs/  
│   └── config.py                # Hyperparameters and settings  
├── results/  
│   └── model.pth                # Trained Model   
│   └── training_progress.png    # Plot visualization  
└── main.py                      # Entry point  
```

# Flappy Bird 
Colab notebook that showcases a custom implementation of a Dueling Deep Q-Network (Dueling DQN) to train an agent that masters the game of Flappy Bird.  
### Results  




https://github.com/user-attachments/assets/9a2e1ed7-6c5a-4f9e-8939-9092ff1f051b





