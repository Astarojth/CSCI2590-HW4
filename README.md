# NLP HW4

This repository mirror contains only the teacher-facing code for Homework 4.

## Structure
- `part1/`: BERT sentiment fine-tuning, OOD transformation, and augmentation code.
- `part2/`: T5 text-to-SQL training, evaluation, inference, and Q4 statistics script.

## Environments
- Part 1: Python 3.10 with `part1/requirements.txt`
- Part 2: Python 3.10 with `part2/requirements.txt`

## Expected Commands
Run commands from the corresponding subdirectory.

### Part 1
- `python3 main.py --train --eval --debug_train`
- `python3 main.py --train --eval`
- `python3 main.py --eval_transformed --debug_transformation`
- `python3 main.py --eval_transformed`
- `python3 main.py --train_augmented --eval_transformed`
- `python3 main.py --eval --model_dir ./out_augmented`
- `python3 main.py --eval_transformed --model_dir ./out_augmented`

### Part 2
- Fine-tuning:
  - `python3 train_t5.py --finetune --learning_rate 1e-4 --scheduler_type linear --num_warmup_epochs 1 --max_n_epochs 20 --patience_epochs 3 --batch_size 2 --test_batch_size 2 --num_beams 4 --experiment_name experiment_bs2`
- From scratch:
  - `python3 train_t5.py --learning_rate 1e-3 --scheduler_type linear --num_warmup_epochs 2 --max_n_epochs 60 --patience_epochs 8 --batch_size 2 --test_batch_size 2 --num_beams 4 --experiment_name experiment_ec_bs2 --submission_name t5_ft_experiment_ec`
- Dev evaluation format check:
  - `python3 evaluate.py --predicted_sql results/t5_ft_experiment_dev.sql --predicted_records records/t5_ft_experiment_dev.pkl --development_sql data/dev.sql --development_records records/ground_truth_dev.pkl`
- Q4 stats:
  - `python3 compute_q4_stats.py --format markdown`

Starter data, model checkpoints, report files, caches, and local experiment artifacts are intentionally excluded.

Final fine-tuned dev artifacts are standardized as `results/t5_ft_experiment_dev.sql` and `records/t5_ft_experiment_dev.pkl`.
