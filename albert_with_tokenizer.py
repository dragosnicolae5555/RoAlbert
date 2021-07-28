# -*- coding: utf-8 -*-
"""ALBERT with Tokenizer
"""

# declare imports
import sentencepiece as sp
import transformers
import shutil
import os
import glob
from transformers import AlbertTokenizer, AlbertConfig, AlbertForMaskedLM, DataCollatorForLanguageModeling, SchedulerType

# specify the path to your language data
token_input = './corpus/merged/train.txt'  # for tokenizer if needed
model_input = './corpus/merged/train.txt' # for ALBERT
# specify the path to tokenizer and model data
model_dir = 'model'
# specify the path to logs
log_dir = 'log'

shutil.rmtree(model_dir, ignore_errors=True)
shutil.rmtree(log_dir, ignore_errors=True)
# creating output dir with tokenizer data
os.mkdir(model_dir)


# Create config for ALBERT, config will be saved to config.json in output dir
def create_config(vocab_size, model_dir):
    # Initializing an ALBERT-base style configuration
    albert_config = AlbertConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072
    )
    albert_config.save_pretrained(model_dir)
    return albert_config


# Creating tokenizer for ALBERT
# Tokenizer is trained on some language data
# cased is used for up and low case. if cased=True then all will be converted to low case
def create_tokenizer_with_training(input, vocab_size, model_dir, cased):
    print("Create tokenizer with training")
    prefix = 'spiece'
    # creating BPE tokenizer for ALBERT model
    # character_coverage=0.9995 for languages with rich character set like Japanese or Chinese
    # you can use user_defined_symbols=[] to define special symbols for tokenizer, i.e. user_defined_symbols=['foo', 'bar']
    sp.SentencePieceTrainer.train(input=glob.glob(input),
                                  model_prefix=prefix,
                                  vocab_size=vocab_size,
                                  model_type='bpe',
                                  character_coverage=0.9995,
                                  hard_vocab_limit=False)

    shutil.move(f'{prefix}.model', f'{model_dir}/{prefix}.model')
    shutil.move(f'{prefix}.vocab', f'{model_dir}/{prefix}.vocab')

    # Creating tokenizer for ALBERT from pretrained
    albert_tokenizer_pretrained = AlbertTokenizer.from_pretrained(model_dir)

    albert_tokenizer_pretrained.do_lower_case = cased

    # save vocabulary of pretrained tokenizer into vocab-{prefix}.model
    albert_tokenizer_pretrained.save_vocabulary(save_directory='.', filename_prefix='vocab')

    return albert_tokenizer_pretrained


# Creating tokenizer for ALBERT from vocabulary
# cased is used for up and low case. if cased=True then all will be converted to low case
def create_tokenizer_from_vocabulary(cased, vocab_file):
    print("create tokenizer")
    # Creating tokenizer for ALBERT from the scratch
    # creating BPE tokenizer for ALBERT model
    # character_coverage=0.9995 for languages with rich character set like Japanese or Chinese
    # you can use user_defined_symbols=[] to define special symbols for tokenizer, i.e. user_defined_symbols=['foo', 'bar']
    return AlbertTokenizer(vocab_file=vocab_file,
                           do_lower_case=cased, model_type='bpe',
                           character_coverage=0.9995,
                           hard_vocab_limit=False)


# Method for dataset getting from input file
def get_dataset(tokenizer, input, block_size):
    from transformers import LineByLineTextDataset
    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=input,
        block_size=block_size,
    )


# Training model with required parameters
def albert_train(model, dataset, batch_size, steps, num_warmup_steps, data_collator, save_steps, learning_rate, output_dir):
    print("Start training")
    from transformers import Trainer, TrainingArguments, IntervalStrategy
# training config
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        max_steps=steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        prediction_loss_only=True,
        logging_dir=f'log/steps_{steps}',
        logging_strategy=IntervalStrategy.EPOCH
    )
# LAMB optimizer
    import torch_optimizer as optim
    optimizer = optim.Lamb(model.parameters(), lr=learning_rate)
    scheduler = transformers.get_scheduler(SchedulerType.LINEAR, optimizer, num_warmup_steps=num_warmup_steps,
                                           num_training_steps=steps)
# trainer creating
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        optimizers=(optimizer, scheduler)
    )
# training
    trainer.train()


# main function to run ALBERT with parameters
# token_input - language data for tokenizer
# model_input - language data for ALBERT
# vocab_size - vocabulary size
# model_dir - model dir
# cased - cased or not cased
# pretrained - train tokenizer or not
# vocab_file - path to vocabulary file
def albert_run(token_input, model_input, vocab_size, model_dir, cased, pretrained, vocab_file=None):
    print("Start run")
    # config creating
    albert_config = create_config(vocab_size, model_dir)
    # model creating
    model = AlbertForMaskedLM(config=albert_config)
    # tokenizer creating
    tokenizer = create_tokenizer_with_training(token_input, vocab_size, model_dir, cased) if pretrained \
        else create_tokenizer_from_vocabulary(cased, vocab_file)
    # data collator creating
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # parameters definition
    save_steps = 2000
    learning_rate = 1e-4

    # first run 900k steps
    #steps = int(9e5)
    steps = 10
    batch_size = 140
    block_size = 128
    num_warmup_steps = steps * 1e-2
    albert_train(model, get_dataset(tokenizer, model_input, block_size), batch_size, steps, num_warmup_steps,
                 data_collator, save_steps, learning_rate, model_dir)

    # second run 100k steps
    #steps = int(1e5)
    steps = 10
    batch_size = 20
    block_size = 512
    num_warmup_steps = steps * 1e-2
    albert_train(model, get_dataset(tokenizer, model_input, block_size), batch_size, steps, num_warmup_steps,
                 data_collator, save_steps, learning_rate, model_dir)


# run model ALBERT with required parameters and tokenizer
# switch cased (True) or uncased (False) on
# use albert_tokenizer_pretrained (pretrained=True) or albert_tokenizer (pretrained=False)
vocab_size = 50000
albert_run(token_input=token_input, model_input=model_input, vocab_size=vocab_size, model_dir=model_dir, cased=True,
           pretrained=False, vocab_file='./vocab-spiece.model')

# to see tensorboard
# - tensorboard --logdir log
#- browse at http://localhost:6006/