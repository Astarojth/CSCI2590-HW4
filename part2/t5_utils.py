import os
import math

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config, Adafactor
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    wandb.init(
        project="nlp-hw4-t5",
        name=args.experiment_name,
        config=vars(args),
    )

def initialize_model(args, tokenizer=None):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    else:
        config = T5Config.from_pretrained("google-t5/t5-small")
        config.decoder_start_token_id = config.pad_token_id
        model = T5ForConditionalGeneration(config)

    if tokenizer is not None and len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.config.decoder_start_token_id = model.config.pad_token_id
    return model.to(DEVICE)

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    checkpoint_path = os.path.join(checkpoint_dir, "best" if best else "last")
    mkdir(checkpoint_path)
    model.save_pretrained(checkpoint_path)

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    model_type = 'ft' if args.finetune else 'scr'
    source_experiment_name = getattr(args, "resume_experiment_name", None) or args.experiment_name
    checkpoint_dir = os.path.join(ROOT_DIR, 'checkpoints', f'{model_type}_experiments', source_experiment_name)
    checkpoint_path = os.path.join(checkpoint_dir, "best" if best else "last")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    model.config.decoder_start_token_id = model.config.pad_token_id
    return model.to(DEVICE)

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=1e-8,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    elif args.optimizer_type == "Adafactor":
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            clip_threshold=1.0,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    accumulation_steps = max(getattr(args, "gradient_accumulation_steps", 1), 1)
    effective_epoch_length = math.ceil(epoch_length / accumulation_steps)
    num_training_steps = effective_epoch_length * args.max_n_epochs
    num_warmup_steps = args.num_warmup_steps if getattr(args, "num_warmup_steps", 0) > 0 else effective_epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
