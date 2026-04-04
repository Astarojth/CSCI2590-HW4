import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data, normalize_sql, repair_predicted_sql, decode_sql_sequences
from utils import compute_metrics, save_queries_and_records, set_random_seeds

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW", "Adafactor"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of micro-batches to accumulate before each optimizer step")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="Gradient clipping norm applied before each optimizer step")

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--num_warmup_steps', type=int, default=0,
                        help="Warmup steps on optimizer updates; overrides num_warmup_epochs when > 0")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")
    parser.add_argument('--full_eval_every_epochs', type=int, default=1,
                        help="Run full generation + SQL/record evaluation every N epochs during training")
    parser.add_argument('--selection_metric', type=str, default='record_f1', choices=['record_f1', 'dev_loss'],
                        help="Metric used for best-checkpoint selection and early stopping")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--max_generation_tokens', type=int, default=256)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--submission_name', type=str, default=None,
                        help="Optional non-canonical test export stem without _test/.sql suffix.")
    parser.add_argument('--skip_test_export', action='store_true',
                        help="Skip final test export during tuning runs")
    parser.add_argument('--augment_sql_vocab', action='store_true',
                        help="Augment the tokenizer with SQL identifier tokens from the training set")
    parser.add_argument('--resume_from_last', action='store_true',
                        help="Resume training from the last checkpoint instead of reinitializing the model")
    parser.add_argument('--resume_from_best', action='store_true',
                        help="Resume training from the best checkpoint instead of reinitializing the model")
    parser.add_argument('--resume_experiment_name', type=str, default=None,
                        help="Optional experiment name to load checkpoints from when resuming")

    args = parser.parse_args()
    return args


def get_test_export_stem(args):
    reserved_submission_stems = {
        "t5_ft_experiment",
        "t5_ft_experiment_ec",
    }
    if args.submission_name is not None:
        if args.submission_name in reserved_submission_stems:
            raise ValueError(
                f"submission_name={args.submission_name} is reserved for canonical submissions; "
                "use autotune_scratch_supervisor.sh to create final submission files."
            )
        return args.submission_name

    model_type = 'ft' if args.finetune else 'scr'
    return f't5_{model_type}_{args.experiment_name}_export'

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_score = None
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join(ROOT_DIR, 'checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path = os.path.join(ROOT_DIR, 'data', 'dev.sql')
    gt_record_path = os.path.join(ROOT_DIR, 'records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join(ROOT_DIR, 'results', f't5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(ROOT_DIR, 'records', f't5_{model_type}_{args.experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss = eval_dev_loss(model, dev_loader)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}")

        record_f1 = None
        record_em = None
        sql_em = None
        error_rate = None
        should_run_full_eval = (
            args.full_eval_every_epochs <= 1
            or epoch % args.full_eval_every_epochs == 0
            or epoch == args.max_n_epochs - 1
        )
        if should_run_full_eval:
            _, record_f1, record_em, sql_em, error_rate = eval_epoch(
                args,
                model,
                dev_loader,
                gt_sql_path,
                model_sql_path,
                gt_record_path,
                model_record_path,
            )
            print(
                f"Epoch {epoch}: Full dev metrics: Record F1: {record_f1}, "
                f"Record EM: {record_em}, SQL EM: {sql_em}"
            )
            print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
            }
            if record_f1 is not None:
                result_dict.update({
                    'dev/record_f1' : record_f1,
                    'dev/record_em' : record_em,
                    'dev/sql_em' : sql_em,
                    'dev/error_rate' : error_rate,
                })
            wandb.log(result_dict, step=epoch)

        current_score = record_f1 if args.selection_metric == 'record_f1' else eval_loss
        if best_score is None:
            improved = True
        elif args.selection_metric == 'record_f1':
            improved = current_score is not None and current_score > best_score
        else:
            improved = current_score < best_score

        if improved:
            best_score = current_score
            epochs_since_improvement = 0
            print(f"Epoch {epoch}: New best checkpoint on {args.selection_metric} = {current_score}")
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if improved:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    accumulation_steps = max(args.gradient_accumulation_steps, 1)

    optimizer.zero_grad()

    for step, (encoder_input, encoder_mask, decoder_input, decoder_targets, _) in enumerate(tqdm(train_loader), start=1):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        loss = criterion(logits.transpose(1, 2), decoder_targets)
        (loss / accumulation_steps).backward()

        should_step = (step % accumulation_steps == 0) or (step == len(train_loader))
        if should_step:
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            non_pad = decoder_targets != PAD_IDX
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens

def eval_dev_loss(model, dev_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        with torch.no_grad():
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']
            loss = criterion(logits.transpose(1, 2), decoder_targets)

        non_pad = decoder_targets != PAD_IDX
        num_tokens = torch.sum(non_pad).item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    predicted_queries = []
    generation_config = GenerationConfig(
        max_new_tokens=args.max_generation_tokens,
        num_beams=args.num_beams,
        early_stopping=args.num_beams > 1,
        decoder_start_token_id=model.config.pad_token_id,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        do_sample=args.do_sample,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        with torch.no_grad():
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']
            loss = criterion(logits.transpose(1, 2), decoder_targets)
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                generation_config=generation_config,
            )

        non_pad = decoder_targets != PAD_IDX
        num_tokens = torch.sum(non_pad).item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        batch_queries = (
            decode_sql_sequences(model.tokenizer, generated, augment_sql_vocab=args.augment_sql_vocab)
            if hasattr(model, "tokenizer") else None
        )
        if batch_queries is None:
            raise RuntimeError("Tokenizer should be attached to the model before evaluation.")
        predicted_queries.extend([repair_predicted_sql(query) for query in batch_queries])

    avg_loss = total_loss / max(total_tokens, 1)
    save_queries_and_records(predicted_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth,
        model_sql_path,
        gt_record_path,
        model_record_path,
    )
    error_rate = sum(bool(msg) for msg in error_msgs) / max(len(error_msgs), 1)
    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    generation_config = GenerationConfig(
        max_new_tokens=args.max_generation_tokens,
        num_beams=args.num_beams,
        early_stopping=args.num_beams > 1,
        decoder_start_token_id=model.config.pad_token_id,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        do_sample=args.do_sample,
        top_p=args.top_p,
        temperature=args.temperature,
    )
    predicted_queries = []

    for encoder_input, encoder_mask, _ in tqdm(test_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                generation_config=generation_config,
            )

        batch_queries = (
            decode_sql_sequences(model.tokenizer, generated, augment_sql_vocab=args.augment_sql_vocab)
            if hasattr(model, "tokenizer") else None
        )
        if batch_queries is None:
            raise RuntimeError("Tokenizer should be attached to the model before inference.")
        predicted_queries.extend([repair_predicted_sql(query) for query in batch_queries])

    save_queries_and_records(predicted_queries, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()
    set_random_seeds(args.seed)
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size,
        args.test_batch_size,
        augment_sql_vocab=args.augment_sql_vocab,
    )
    if args.resume_from_last or args.resume_from_best:
        model = load_model_from_checkpoint(args, best=args.resume_from_best)
        if len(train_loader.dataset.tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(train_loader.dataset.tokenizer))
    else:
        model = initialize_model(args, tokenizer=train_loader.dataset.tokenizer)
    model.tokenizer = train_loader.dataset.tokenizer
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    args.resume_experiment_name = None
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    model.tokenizer = dev_loader.dataset.tokenizer
    gt_sql_path = os.path.join(ROOT_DIR, 'data', 'dev.sql')
    gt_record_path = os.path.join(ROOT_DIR, 'records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join(ROOT_DIR, 'results', f't5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(ROOT_DIR, 'records', f't5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args,
        model,
        dev_loader,
        gt_sql_path,
        model_sql_path,
        gt_record_path,
        model_record_path,
    )
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    if not args.skip_test_export:
        model.tokenizer = test_loader.dataset.tokenizer
        submission_stem = get_test_export_stem(args)
        model_sql_path = os.path.join(ROOT_DIR, 'results', f'{submission_stem}_test.sql')
        model_record_path = os.path.join(ROOT_DIR, 'records', f'{submission_stem}_test.pkl')
        test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
