# Reward-Modeling-Human-Preferences-Actor-Critic
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