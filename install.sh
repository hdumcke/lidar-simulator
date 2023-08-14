#!/bin/bash

pip install -r requirements.txt
ipython kernel install --user --name=lidar_sim
pip install -e python/lidar_sim
pip install -e python/utilities
pip install -e python/hough_transform
pip install -e python/split_and_merge
pip install -e python/ransac
pip install -e python/incremental
pip install -e python/lineregression
