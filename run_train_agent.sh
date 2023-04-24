#!/bin/bash
python train_agent.py \
--data_path /home/linjie/projects/KG/PGPR_hetionet/data \
--dataset hetionet \
--output_dir /home/linjie/projects/KG/PGPR_hetionet/output \
--epochs 50 \
--batch_size 1024 \
--max_acts 100 \
--max_path_len 4 \
--head_entity Compound \
--tail_entity Compound \
--relation_type CmC