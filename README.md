# Reward-Modeling-Human-Preferences-Actor-Critic

David Dunlap and Ian Randman

Reward Modeling from Human Preferences and Advantage Actor-Critic Reinforcement Learning: A Reproducibility Study

Some dependencies are specific to the OS. Please review the list below to see the list of appropriate commands.
Must install Anaconda and ffmpeg. 
```
brew cask install anaconda (for MacOS)
conda create -n reward-modeling python=3.7 pip
pip install flask
pip install keras
pip install scikit-learn
pip install tensorflow==1.15
pip install gym
pip install gym[atari] OR pip install git+https://github.com/Kojoley/atari-py.git (for Windows) 
conda install -c conda-forge swig
pip install matplotlib
pip install Box2D
brew install ffmpeg (for MacOS)
```

The entry point to the program is in `main.py`. To change which environments are tested, modify the env_lst list. To
change some overall specifics about the run, modify the parameters of TrainingSystem in main.py. The record parameter
specifies if the runs are recorded (mandatory for giving human feedback), use_reward_model specifies whether to use the
learned reward model or the OpenAI provided reward function, and load_model specifies whether or not to load a saved
model file.

To give feedback to the model, run the program and go to `localhost:5000`.