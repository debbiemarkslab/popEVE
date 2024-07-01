#!/bin/bash
# Example installation script of popEVE for pytorch-cuda from scratch
# Tested on Ubuntu Server 24.04 LTS, runtime ~10 minutes including a reboot.
# Before running this script, first run `git clone https://github.com/debbiemarkslab/popEVE`
# and then `cd popEVE`
# If NVIDIA drivers have not been installed before, this script must be run twice, rebooting the system in between.

# install NVIDIA drivers
if [ ! -f "/proc/driver/nvidia/version" ]; then
  echo "NVIDIA driver not found; installing."
  if sudo apt update && sudo apt install -y --no-install-recommends nvidia-driver-535; then
    echo "
    NVIDIA drivers installed.
    Please reboot your system, then run linux_setup.sh a second time."
  else
    echo "NVIDIA driver install failed; exiting."
  fi
  exit
fi

# set up conda
if [ ! -d "$HOME/miniconda3" ]; then
  echo "miniconda3 not found; installing."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  sh Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME"/miniconda3
  rm Miniconda3-latest-Linux-x86_64.sh
fi
"$HOME"/miniconda3/bin/conda init

# download popEVE code:
# git clone https://github.com/debbiemarkslab/popEVE
# cd popEVE

# create conda environment
"$HOME"/miniconda3/bin/conda env create -f popeve_env_linux.yml

echo "
popEVE installed.
Run 'source ~/.bashrc; conda activate popeve_env' before using."

# # to train popEVE models:
# ./train_popEVE_models.sh
