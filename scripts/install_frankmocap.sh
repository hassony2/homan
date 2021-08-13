#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

echo ""
echo ">>  Installing a third-party 2D keypoint detector"
sh external/frankmocap/scripts/install_pose2d.sh

echo ""
echo ">>  Download extra data for body module"
sh external/frankmocap/scripts/download_data_body_module.sh


echo ""
echo ">>  Installing a third-party hand detector"
sh external/frankmocap/scripts/install_hand_detectors.sh


echo ""
echo ">>  Download extra data for hand module"
sh external/frankmocap/scripts/download_data_hand_module.sh
