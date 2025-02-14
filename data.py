import functools
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile

import multiprocessing
import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers

import logging
import lightning

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

image_data_name_dict = {
    "CIFAR10":CIFAR10, 
    "MNIST":MNIST
}

text_data_name_dict = {
    "lm1b": None
}
protein_data_name_dict = {
    "uniref50": None
}

def fsspec_exists(filename):
  """Check if a file exists using fsspec."""
  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def fsspec_listdir(dirname):
  """Listdir in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
  """Mkdirs in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
  if torch.isnan(tensor).any():
    print(name, tensor)


class LoggingContext:
  """Context manager for selective logging."""
  def __init__(self, logger, level=None, handler=None, close=True):
    self.logger = logger
    self.level = level
    self.handler = handler
    self.close = close

  def __enter__(self):
    if self.level is not None:
      self.old_level = self.logger.level
      self.logger.setLevel(self.level)
    if self.handler:
      self.logger.addHandler(self.handler)

  def __exit__(self, et, ev, tb):
    if self.level is not None:
      self.logger.setLevel(self.old_level)
    if self.handler:
      self.logger.removeHandler(self.handler)
    if self.handler and self.close:
      self.handler.close()


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
  """Initializes multi-GPU-friendly python logger."""

  logger = logging.getLogger(name)
  logger.setLevel(level)

  # this ensures all logging levels get marked with the rank zero decorator
  # otherwise logs would get multiplied for each GPU process in multi-GPU setup
  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(
              getattr(logger, level)))

  return logger

LOGGER = get_logger(__name__)


def wt_detokenizer(string):
  # contractions
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  # number separators
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  # punctuation
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  # double brackets
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  # miscellaneous
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")
  return string


def ptb_detokenizer(x):
  x = x.replace(" 's", "'s")
  x = x.replace("s ' ", "s' ")
  x = x.replace(" n't", "n't")
  x = x.replace(" \n ", "\n")
  x = x.replace("\\/", "/")
  for _ in range(10):
      x = x.replace(" N ", " 1 ")
  x = x.replace("$ 1", "$1")
  x = x.replace("# 1", "#1")
  x = x.replace("<unk>", "?")
  return x


def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x


def lambada_detokenizer(text):
  text = text.replace("“", '"')
  text = text.replace("”", '"')
  return '\n'+text.strip()


def scientific_papers_detokenizer(x):
  x = wt_detokenizer(x)
  x = lm1b_detokenizer(x)
  return x


class Text8Tokenizer(transformers.PreTrainedTokenizer):
  def __init__(
    self,
    bos_token='[BOS]',
    eos_token='[EOS]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    pad_token='[PAD]',
    mask_token='[MASK]',
    unk_token='[UNK]',
    **kwargs):
    self.characters = list('abcdefghijklmnopqrstuvwxyz ')
    self._vocab_str_to_int = {
      '[CLS]': 0,
      '[SEP]': 1,
      '[BOS]': 2,
      '[EOS]': 3,
      '[MASK]': 4,
      '[PAD]': 5,
      '[RESERVED]': 6,
      '[UNK]': 7,
      ** {ch: i + 8 for i, ch in enumerate(self.characters)}}
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> typing.Dict[str, int]:
    return self._vocab_str_to_int


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
      response = requests.get(url, stream=True)
      data_list = []

      # Process each line in the response content
      for line in response.iter_lines(decode_unicode=True):
        if line:
          data = json.loads(line)
          data_list.append(data)

      return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset

def get_text8_dataset(cache_dir, max_seq_length=256,
                      drop_last=True, crop_train=False):
  """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
  """
  url = 'http://mattmahoney.net/dc/text8.zip'
  if not crop_train:
    cache_dir = f'{cache_dir}/text8'
  else:
    cache_dir = f'{cache_dir}/text8-crop-train'
  split_names = ['train', 'validation', 'test']
  if not all([
    fsspec_exists(os.path.join(cache_dir, split))
    for split in split_names
  ]):
    # Check if raw data exists
    raw_cache_dir = os.path.join(cache_dir, 'raw_data')
    if not all([
      fsspec_exists(
        os.path.join(raw_cache_dir, f'text8.{split}.txt'))
      for split in split_names
    ]):
      if not fsspec_exists(
        os.path.join(raw_cache_dir, 'text8.zip')):
        fsspec_mkdirs(raw_cache_dir, exist_ok=True)
        LOGGER.info('Downloading text8 from URL {}.'.format(url))
        with urllib.request.urlopen(url) as in_stream:
          with open(os.path.join(raw_cache_dir, 'text8.zip'), 'wb') as out_file:
            shutil.copyfileobj(in_stream, out_file)

      with fsspec.open(
        os.path.join(raw_cache_dir, 'text8.zip'),
        'rb') as f:
        rawdata = zipfile.ZipFile(f).read(
          'text8').decode('utf-8')

      # Splits taken from D3PM codebase
      splits = {
        'train': rawdata[:90000000],
        'validation': rawdata[90000000: 95000000],
        'test': rawdata[95000000:],
      }

      for split, data in splits.items():
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'w') as f:
          f.write(data)
    else:
      splits = {}
      for split in split_names:
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'r') as f:
          splits[split] = f.read()

    # Chunk and save as datasets.DatasetDict
    def chunks(lst, n):
      """Yield successive n-sized chunks from lst."""
      for i in range(0, len(lst), n):
        yield lst[i:i + n]

    dataset_dict = {}
    for k, v in splits.items():
      if k == 'train' and crop_train == True:
        chunk_size = 2 * max_seq_length
      else:
        chunk_size = max_seq_length
      text = list(chunks(v, chunk_size))
      if drop_last and len(text[-1]) < chunk_size:
        text = text[:-1]
      dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(cache_dir)
  else:
    dataset = datasets.load_from_disk(cache_dir)

  return dataset


def _group_texts(examples, block_size, bos, eos):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    _values.append(
      [bos]
      + concatenated_examples[i : i + new_block_size]
      + [eos])
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result


def get_dataset(
    dataset_name, tokenizer, wrap, mode, cache_dir,
    block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False):
  if wrap:
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped.dat'
  else:
    filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped.dat'
  _path = os.path.join(cache_dir, filename)
  
  if fsspec_exists(_path):
    LOGGER.info(f'Loading data from: {_path}')
    return datasets.load_from_disk(_path).with_format('torch')
  LOGGER.info(f'Generating new data at: {_path}')

  crop_train = dataset_name == 'text8-crop'
  if mode == 'train' and crop_train:
    # double block size for sub-sampling
    block_size *= 2
  
  if dataset_name == 'wikitext103':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-103-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'wikitext2':
    dataset = datasets.load_dataset(
      'wikitext',
      name='wikitext-2-raw-v1',
      cache_dir=cache_dir)
  elif dataset_name == 'ptb':
    dataset = datasets.load_dataset(
      'ptb_text_only', cache_dir=cache_dir)
  elif dataset_name == 'lambada':
    dataset = get_lambada_test_dataset()
  elif dataset_name == 'text8':
    assert wrap
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size)
  elif dataset_name == 'text8-crop':
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size, crop_train=True)
  elif dataset_name == 'openwebtext-train':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[:-100000]',
      cache_dir=cache_dir,
      streaming=streaming, trust_remote_code=True)
  elif dataset_name == 'openwebtext-valid':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[-100000:]',
      cache_dir=cache_dir,
      streaming=streaming, trust_remote_code=True)
  elif dataset_name == 'scientific_papers_arxiv':
    dataset = datasets.load_dataset(
      'scientific_papers', 'arxiv',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'scientific_papers_pubmed':
    dataset = datasets.load_dataset(
      'scientific_papers', 'pubmed',
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming)
  elif dataset_name == 'ag_news':
    dataset = datasets.load_dataset(
      'ag_news',
      cache_dir=cache_dir,
      streaming=streaming, trust_remote_code=True)
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming, trust_remote_code=True)

  if dataset_name in ['lambada', 'openwebtext-train',
                      'openwebtext-valid']:
    data = dataset
  else:
    data = dataset[mode]

  if dataset_name.startswith('wikitext'):
    detokenizer = wt_detokenizer
  elif dataset_name == 'ptb':
    detokenizer = ptb_detokenizer
  elif dataset_name == 'lm1b':
    detokenizer = lm1b_detokenizer
  elif dataset_name == 'lambada':
    detokenizer = lambada_detokenizer
  elif dataset_name.startswith('scientific_papers'):
    detokenizer = scientific_papers_detokenizer
  else:
    detokenizer = None

  def _apply_detokenizer(detokenizer):
    def detok(text):
      for i, t in enumerate(text, 0):
        text[i] = detokenizer(t)
      return text
    return detok
  
  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  def preprocess_and_tokenize(example):
    if dataset_name == 'ptb':
      text = example['sentence']
    elif 'scientific_papers' in dataset_name:
      text = example['article']
    else:
      text = example['text']
    
    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    if wrap:
      tokens = tokenizer(text,
                         add_special_tokens=False,
                         return_attention_mask=False,
                         return_token_type_ids=False)
      tokens = {'input_ids':
                [t + [EOS] for t in tokens['input_ids']]}
      # Still missing BOS, but will be added in group_texts
    else:
      tokens = tokenizer(text,
                         max_length=block_size,
                         padding='max_length',
                         truncation=True,
                         add_special_tokens=True,
                         return_attention_mask=True,
                         return_token_type_ids=True)
    return tokens

  if streaming:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      desc='Tokenizing')
  else:
    tokenized_dataset = data.map(
      preprocess_and_tokenize,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Tokenizing')
  if dataset_name == 'ptb':
    tokenized_dataset = tokenized_dataset.remove_columns(
      'sentence')
  elif 'scientific_papers' in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns([
      'article', 'abstract', 'section_names'])
  elif dataset_name == 'ag_news':
    tokenized_dataset = tokenized_dataset.remove_columns(
      ['text', 'label'])
  else:
    tokenized_dataset = tokenized_dataset.remove_columns(
      'text')

  if not wrap:
    tokenized_dataset.save_to_disk(_path)
    return tokenized_dataset.with_format('torch')

  group_texts = functools.partial(
    _group_texts, block_size=block_size, bos=BOS, eos=EOS)
  if streaming:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      desc='Grouping')
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc='Grouping')
    chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset


def get_tokenizer(config):
  if config.data.tokenizer_name_or_path == 'text8':
    tokenizer = Text8Tokenizer()
  elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
    tokenizer = transformers.BertTokenizer.\
      from_pretrained('bert-base-uncased')
  elif config.data.tokenizer_name_or_path == 'gpt2':
    tokenizer = transformers.GPT2TokenizerFast.\
      from_pretrained('gpt2')
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.data.tokenizer_name_or_path)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        'Tokenizer must have a bos_token or '
        f'cls_token: {tokenizer}')
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer
    

def get_text_dataloaders(
    config, 
    tokenizer, 
    skip_train=False,
    skip_valid=False, 
    valid_seed=None
):
  num_gpus = torch.cuda.device_count()
#   assert (config.loader.global_batch_size
#           == (config.loader.batch_size
#               * config.trainer.num_nodes
#               * num_gpus
#               * config.trainer.accumulate_grad_batches))
  
#   if config.loader.global_batch_size % (
#     num_gpus * config.trainer.accumulate_grad_batches) != 0:
#     raise ValueError(
#       f'Train Batch Size {config.training.batch_size}'
#       f'not divisible by {num_gpus} gpus with accumulation '
#       f'{config.trainer.accumulate_grad_batches}.')
    
#   if config.loader.eval_global_batch_size % num_gpus != 0:
#     raise ValueError(
#       f'Eval Batch Size for {config.eval.batch_size} '
#       f'not divisible by {num_gpus}.')
    
  if skip_train:
    train_set = None
  else:
    train_set = get_dataset(
      config.data.data,
      tokenizer,
      mode='train',
      wrap=config.data.wrap,
      cache_dir=config.data.cache_dir,
      # block_size=config.model.length
    )
  
  if config.data.data in ['text8', 'lm1b', 'ag_news']:
    validation_split = 'test'
  else:
    validation_split = 'validation'
  if skip_valid:
    valid_set = None
  else:
    valid_set = get_dataset(
      config.data.valid,
      tokenizer,
      wrap=config.data.wrap,
      mode=validation_split,
      cache_dir=config.data.cache_dir,
    #   block_size=config.model.length,
      streaming=False
    )

    # multiprocessing.cpu_count()
  num_workers = 16//max([1, torch.cuda.device_count()])

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.train.batch_size,
      num_workers=num_workers,
    #   pin_memory=config.loader.pin_memory,
    #   shuffle=not config.data.streaming,
      persistent_workers=True
    )
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.train.batch_size,
      num_workers=num_workers,
    #   pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator
    )
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader

def get_img_dataloaders(cfg):
    batch_size = cfg.train.batch_size
        
    train_dataset = image_data_name_dict[cfg.data.data](
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
    test_dataset = image_data_name_dict[cfg.data.data](
        "./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
    
    def collate_fn(batch):
        x, cond = zip(*batch)
        x = torch.stack(x)
        cond = torch.tensor(cond)
        cond = cond * cfg.data.conditional
        x = (x * (cfg.data.N - 1)).round().long().clamp(0, cfg.data.N - 1)
        return x, cond
    
    # train_size = int(len(full_dataset) * 0.9)
    # train_dataset, test_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    # multiprocessing.cpu_count()
    num_workers = 16//max([1, torch.cuda.device_count()])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_dataloader, test_dataloader

from sequence_models.datasets import UniRefDataset
from evodiff.utils import Tokenizer
from d3pm_sc.utils import _pad 
import numpy as np
def get_protein_dataloaders(cfg):
    batch_size = cfg.train.batch_size

    max_len = 1024
    tokenizer = Tokenizer()
    print("Getting Uniref.")
    train_dataset = UniRefDataset('/scratch/aa11803/d3pm/data/uniref_2020/uniref50/', 'train', structure=False, max_len=max_len)
    test_dataset = UniRefDataset('/scratch/aa11803/d3pm/data/uniref_2020/uniref50/', 'test', structure=False, max_len=max_len)

    def mask_pad(tokenized):
        masks = tokenized != tokenizer.pad_id
        return tokenized.long(), masks.float()
    def collate_fn(batch):
        tokenized = [torch.tensor(tokenizer.tokenize(s)) for s in batch]
        tokenized = _pad(tokenized, tokenizer.pad_id)
        return mask_pad(tokenized)

    # multiprocessing.cpu_count()
    print("Setting N workers.")
    num_workers = 16//torch.cuda.device_count()
    if hasattr(cfg.train, 'pack') and cfg.train.pack:
        block_size = 13
        def collate_fn_pack(batch):
            batch = [string[0] + tokenizer.pad for string in batch]
            if len(batch) % block_size != 0:
                batch = batch + (block_size - len(batch) % block_size) * ['']
            strings = np.array(batch).reshape(-1, block_size)
            strings[0, 1:] = ''
            strings = [(''.join(strs)[:max_len],) for strs in strings]
            return collate_fn(strings)
        print("Building dataloader.")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size* block_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn_pack)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    print("Building test dataloader.")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_dataloader, test_dataloader

def get_dataloaders(cfg):
    if cfg.data.data in image_data_name_dict:
        return get_img_dataloaders(cfg)
    elif cfg.data.data in text_data_name_dict:
        tokenizer = get_tokenizer(cfg) 
        return get_text_dataloaders(cfg, tokenizer)
    elif cfg.data.data in protein_data_name_dict:
        return get_protein_dataloaders(cfg)
    else:
        pass
