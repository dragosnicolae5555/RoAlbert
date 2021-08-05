
# declare imports
import sentencepiece as sp
import transformers
import shutil
import os
import glob
from transformers import AlbertTokenizer, AlbertConfig, AlbertForMaskedLM, DataCollatorForLanguageModeling, \
    SchedulerType
from collections import namedtuple

Checkpoints = namedtuple('CheckPoints', 'use_checkpoint first_run_checkpoint_dir second_run_checkpoint_dir')

# used by the CUDA driver to decide what devices should be visible to CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["LD_PRELOAD"] = "/usr/lib/libtcmalloc_minimal.so.4"
# specify the path to your language data
token_input = './train.txt'  # for tokenizer if needed
model_input = './valid.txt'  # for ALBERT
# specify the path to tokenizer and model data
model_dir = 'model'
# specify the path to logs
log_dir = 'log'


def remove_log_and_model_dir():
    shutil.rmtree(model_dir, ignore_errors=True)
    shutil.rmtree(log_dir, ignore_errors=True)


# Create config for ALBERT, config will be safed to config.json in output dir
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
    # creating BPE tokenizer for ALBERT model
    # character_coverage=0.9995 for languages with rich character set like Japanese or Chinese
    # you can use user_defined_symbols=[] to define special symbols for tokenizer, i.e. user_defined_symbols=['foo', 'bar']
    # use input_sentence_size=<size> to limit the size for training and shuffle_input_sentence=True to shuffle data from input
    sp.SentencePieceTrainer.train(input=glob.glob(input),
                                  model_prefix='spiece',
                                  vocab_size=vocab_size,

                                  shuffle_input_sentence=True,
                                  model_type='bpe',
                                  character_coverage=0.9995,
                                  hard_vocab_limit=False)

    shutil.move('spiece.model', f'{model_dir}/spiece.model')
    shutil.move('spiece.vocab', f'{model_dir}/spiece.vocab')

    # Creating tokenizer for ALBERT from pretrained
    albert_tokenizer_pretrained = AlbertTokenizer.from_pretrained(model_dir)

    albert_tokenizer_pretrained.do_lower_case = cased

    # save vocabulary of pretrained tokenizer into vocab-spiece.model
    albert_tokenizer_pretrained.save_vocabulary(save_directory='.', filename_prefix='vocab')

    return albert_tokenizer_pretrained


# Creating tokenizer for ALBERT from vocabulary
# cased is used for up and low case. if cased=True then all will be converted to low case
def create_tokenizer_from_vocabulary(cased, vocab_file):
    # Creating tokenizer for ALBERT from the scratch
    # creating BPE tokenizer for ALBERT model
    # character_coverage=0.9995 for languages with rich character set like Japanese or Chinese
    # you can use user_defined_symbols=[] to define special symbols for tokenizer, i.e. user_defined_symbols=['foo', 'bar']
    albert_tokenizer = AlbertTokenizer(vocab_file=vocab_file,
                                       do_lower_case=cased, model_type='bpe',
                                       character_coverage=0.9995,
                                       hard_vocab_limit=False)
    return albert_tokenizer


# Method for dataset getting from input dir
def get_dataset(tokenizer, input, block_size):
    from transformers import LineByLineTextDataset
    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=input,
        block_size=block_size,
    )


# Training model with requered parameters
def albert_train(model, dataset, batch_size, steps, num_warmup_steps, data_collator, save_steps, learning_rate,
                 output_dir, logging_steps, resume_from_checkpoint):
    from transformers import Trainer, TrainingArguments, IntervalStrategy
    # trainig config
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        max_steps=steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        prediction_loss_only=True,
        logging_dir=f'log/steps_{steps}',
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=logging_steps
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
    # traning
    trainer.train(resume_from_checkpoint)


# main function to run ALBERT with parameters
# token_input - language data for tokenizer
# model_input - language data for ALBERT
# vocab_size - vocabulary size
# model_dir - model dir
# cased - cased or not cased
# pretrained - train tokenizer or not
# vocab_file - path to vocabulary file
# logging_steps - number of update steps between two logs
# checkpoints - checkpoints to restore
def albert_run(token_input, model_input, vocab_size, model_dir, cased, pretrained, vocab_file=None, logging_steps=500,
               checkpoints=Checkpoints(False, '', '')):
    def get_checkpoint(run_subfolder, use_checkpoint, checkpoint_dir):
        run_subfolder_not_empty = os.path.exists(run_subfolder) and os.path.isdir(run_subfolder) and len(
            os.listdir(run_subfolder)) != 0
        if run_subfolder_not_empty and use_checkpoint:
            checkpoint = True if checkpoint_dir == '' else checkpoint_dir
        else:
            checkpoint = None
        return checkpoint

    # config creating
    albert_config = create_config(vocab_size, model_dir)
    # model creating
    model = AlbertForMaskedLM(config=albert_config)
    # tokenizer creating
    tokenizer = create_tokenizer_with_training(token_input, vocab_size, model_dir,
                                               cased) if pretrained else create_tokenizer_from_vocabulary(cased,
                                                                                                          vocab_file)
    # data collator creating
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # parameters definition
    save_steps = 2000
    learning_rate = 1e-4

    # first run 900k steps
    steps = int(9e5)
    batch_size = 140
    block_size = 128
    run_subfolder = f'{model_dir}/first_run'
    num_warmup_steps = steps * 1e-2
    albert_train(model, get_dataset(tokenizer, model_input, block_size), batch_size, steps, num_warmup_steps,
                 data_collator, save_steps, learning_rate, run_subfolder, logging_steps,
                 get_checkpoint(run_subfolder, checkpoints.use_checkpoint, checkpoints.first_run_checkpoint_dir))

    # second run 100k steps
    steps = int(1e5)
    batch_size = 20
    block_size = 512
    run_subfolder = f'{model_dir}/second_run'
    num_warmup_steps = steps * 1e-2
    checkpoint = get_checkpoint(run_subfolder, checkpoints.use_checkpoint, checkpoints.second_run_checkpoint_dir) if \
        checkpoints.use_checkpoint and (checkpoints.first_run_checkpoint_dir == '' or
                                        checkpoints.first_run_checkpoint_dir != '' and
                                        checkpoints.second_run_checkpoint_dir != '') else None
    albert_train(model, get_dataset(tokenizer, model_input, block_size), batch_size, steps, num_warmup_steps,
                 data_collator, save_steps, learning_rate, run_subfolder, logging_steps, checkpoint)


# run model ALBERT with required parameters and tokenizer
# switch cased (True) or uncased (False) on
# use albert_tokenizer_pretrained (pretrained=True) or albert_tokenizer (pretrained=False)
# to restore from a checkpoint set use_checkpoint = True the last checkpoint from model folder will be used, you can
# set first_run_checkpoint_dir, second_run_checkpoint_dir to use a specific checkpoint (if set
# only first_run_checkpoint_dir the first run will be restored and the second one will not)
vocab_size = 10000
use_checkpoint = False
first_run_checkpoint_dir = ''
second_run_checkpoint_dir = ''

if not use_checkpoint:
    remove_log_and_model_dir()
albert_run(token_input=token_input, model_input=model_input, vocab_size=vocab_size, model_dir=model_dir, cased=True,
           pretrained=False, vocab_file='vocab-spiece.model', logging_steps=5,
           checkpoints=Checkpoints(use_checkpoint, first_run_checkpoint_dir, second_run_checkpoint_dir))