#!/bin/sh
session="handover"
tmux new-session -d -s $session
tmux send-keys 'roslaunch e2e_handover inference.launch'
tmux split-window -h
tmux send-keys 'roslaunch e2e_handover position.launch'
tmux split-window -v
tmux send-keys 'roslaunch e2e_handover papillarray.launch'
tmux split-window -v -t 0
tmux send-keys 'roslaunch e2e_handover recording.launch'
tmux select-pane -t 0
tmux attach-session -d -t $session
export PS1='${debian_chroot:+($debian_chroot)}\w\$ '
