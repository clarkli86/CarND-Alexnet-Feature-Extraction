#!/bin/sh
wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip
unzip traffic-signs-data.zip
# Install opencv
source activate carnd-term1
conda install -c menpo opencv
