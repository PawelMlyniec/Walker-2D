#!/usr/bin/env bash
python3 -m venv env
source env/bin/activate

git clone https://github.com/openai/gym.git
cd gym
pip install -e .
cd ..

git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
cd ..

