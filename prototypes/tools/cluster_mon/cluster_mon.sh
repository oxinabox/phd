#!/usr/bin/env bash
tmux select-layout even-horizontal

tmux split-window -dh 'ssh\ amon'
tmux split-window -dh 'ssh\ heathred'

tmux send-keys -t 2 'htop' enter
tmux send-keys -t 3 'htop' enter
htop

