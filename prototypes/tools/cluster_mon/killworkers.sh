#!/usr/bin/env bash
pkill -f "julia --worker" 
ssh amon 'pkill -f "julia --worker"' 
ssh zeus 'pkill -f  "julia --worker"'
ssh jove 'pkill -f "julia --worker"' 
ssh ares 'pkill -f "julia --worker"' 
ssh heathred 'pkill  -f "julia --worker"' 
