#!/bin/bash

pip install -r requirements.txt
ipython kernel install --user --name=lidar_sim
pip install -e python/lidar_sim
pip install -e python/split_and_merge
