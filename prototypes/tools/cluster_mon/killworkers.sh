#!/usr/bin/env bash
pkill worker
ssh amon pkill worker
ssh zeus pkill worker
ssh jove pkill worker
ssh heathred pkill worker
