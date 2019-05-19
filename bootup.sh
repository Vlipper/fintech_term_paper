#! /bin/bash
set -e

# make session and run tensorboard there
tmux new -d -s tensorboard 'tensorboard --logdir /mntlong/lanl_comp/logs/runs/'

# make train ready session
tmux new -d -s train && tmux send-keys -t train C-z 'cd /mntlong/scripts/lanl_comp/' Enter && tmux a -t train
