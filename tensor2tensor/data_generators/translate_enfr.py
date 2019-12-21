# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import problem, cleaner_en_xx, generator_utils, multi_problem, \
    translate_ende, translate_enro
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.data_generators.oov_text_encoder import OOVTokenTextEncoder
from tensor2tensor.data_generators.text_encoder import EOS_ID
from tensor2tensor.data_generators.text_problems import VocabType, text2text_generate_encoded
from tensor2tensor.data_generators.translate import _preprocess_sgm
from tensor2tensor.data_generators.wiki_lm import LanguagemodelEnWiki32k, _FR_TRAIN_NAME_ID, \
  _EN_TRAIN_NAME_ID, _FR_EVAL_NAME_ID, _EN_EVAL_NAME_ID, _FR_TEST_NAME_ID, _EN_TEST_NAME_ID
from tensor2tensor.models.transformer import transformer_tall_pretrain_lm_tpu_adafactor, \
  transformer_tiny
from tensor2tensor.utils import registry, mlperf_log

import tensorflow as tf

from tensor2tensor.utils.vocab import Vocab

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_ENFR_TRAIN_SMALL_DATA = [
    [
        "https://s3.amazonaws.com/opennmt-trainingdata/baseline-1M-enfr.tgz",
        ("baseline-1M-enfr/baseline-1M_train.en",
         "baseline-1M-enfr/baseline-1M_train.fr")
    ],
]
_ENFR_TEST_SMALL_DATA = [
    [
        "https://s3.amazonaws.com/opennmt-trainingdata/baseline-1M-enfr.tgz",
        ("baseline-1M-enfr/baseline-1M_valid.en",
         "baseline-1M-enfr/baseline-1M_valid.fr")
    ],
]
_ENFR_TRAIN_LARGE_DATA = [
    [
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        ("commoncrawl.fr-en.en", "commoncrawl.fr-en.fr")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        ("training/europarl-v7.fr-en.en", "training/europarl-v7.fr-en.fr")
    ],
    [
        "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz",
        ("training/news-commentary-v9.fr-en.en",
         "training/news-commentary-v9.fr-en.fr")
    ],
    [
        "http://www.statmt.org/wmt10/training-giga-fren.tar",
        ("giga-fren.release2.fixed.en.gz",
         "giga-fren.release2.fixed.fr.gz")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-un.tgz",
        ("un/undoc.2000.fr-en.en", "un/undoc.2000.fr-en.fr")
    ],
]
_ENFR_TEST_LARGE_DATA = [
    [
        "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        ("dev/newstest2013.en", "dev/newstest2013.fr")
    ],
]


@registry.register_problem
class TranslateEnfrWmtSmall8k(translate.TranslateProblem):
  """Problem spec for WMT En-Fr translation."""

  @property
  def approx_vocab_size(self):
    return 2**13  # 8192

  @property
  def use_small_dataset(self):
    return True

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    if self.use_small_dataset:
      datasets = _ENFR_TRAIN_SMALL_DATA if train else _ENFR_TEST_SMALL_DATA
    else:
      datasets = _ENFR_TRAIN_LARGE_DATA if train else _ENFR_TEST_LARGE_DATA
    return datasets

  def vocab_data_files(self):
    return (_ENFR_TRAIN_SMALL_DATA if self.use_small_dataset
            else _ENFR_TRAIN_LARGE_DATA)


@registry.register_problem
class TranslateEnfrWmtSmall32k(TranslateEnfrWmtSmall8k):

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768


@registry.register_problem
class TranslateEnfrWmt8k(TranslateEnfrWmtSmall8k):

  @property
  def use_small_dataset(self):
    return False


@registry.register_problem
class TranslateEnfrWmt32k(TranslateEnfrWmtSmall32k):

  @property
  def use_small_dataset(self):
    return False


@registry.register_problem
class TranslateEnfrWmt32kPacked(TranslateEnfrWmt32k):

  @property
  def packed_length(self):
    return 256

  @property
  def use_vocab_from_other_problem(self):
    return TranslateEnfrWmt32k()


@registry.register_problem
class TranslateEnfrWmt32kWithBacktranslateFr(TranslateEnfrWmt32k):
  """En-Fr translation with added French data, back-translated."""

  @property
  def use_vocab_from_other_problem(self):
    return TranslateEnfrWmt32k()

  @property
  def already_shuffled(self):
    return True

  @property
  def skip_random_fraction_when_training(self):
    return False

  @property
  def backtranslate_data_filenames(self):
    """List of pairs of files with matched back-translated data."""
    # Files must be placed in tmp_dir, each similar size to authentic data.
    return [("fr_mono_en.txt", "fr_mono_fr.txt")]

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 1,  # Use just 1 shard so as to not mix data.
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    datasets = self.source_data_files(dataset_split)
    tag = "train" if dataset_split == problem.DatasetSplit.TRAIN else "dev"
    data_path = translate.compile_data(
        tmp_dir, datasets, "%s-compiled-%s" % (self.name, tag))
    # For eval, use authentic data.
    if dataset_split != problem.DatasetSplit.TRAIN:
      for example in text_problems.text2text_txt_iterator(
          data_path + ".lang1", data_path + ".lang2"):
        yield example
    else:  # For training, mix synthetic and authentic data as follows.
      for (file1, file2) in self.backtranslate_data_filenames:
        path1 = os.path.join(tmp_dir, file1)
        path2 = os.path.join(tmp_dir, file2)
        # Synthetic data first.
        for example in text_problems.text2text_txt_iterator(path1, path2):
          yield example
        # Now authentic data.
        for example in text_problems.text2text_txt_iterator(
            data_path + ".lang1", data_path + ".lang2"):
          yield example


@registry.register_problem
class TranslateEnfrWmt32kWithBacktranslateEn(
    TranslateEnfrWmt32kWithBacktranslateFr):
  """En-Fr translation with added English data, back-translated."""

  @property
  def backtranslate_data_filenames(self):
    """List of pairs of files with matched back-translated data."""
    # Files must be placed in tmp_dir, each similar size to authentic data.
    return [("en_mono_en.txt%d" % i, "en_mono_fr.txt%d" % i) for i in [0, 1, 2]]


@registry.register_problem
class TranslateEnfrWmtSmallCharacters(translate.TranslateProblem):
  """Problem spec for WMT En-Fr translation."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  @property
  def use_small_dataset(self):
    return True

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    if self.use_small_dataset:
      datasets = _ENFR_TRAIN_SMALL_DATA if train else _ENFR_TEST_SMALL_DATA
    else:
      datasets = _ENFR_TRAIN_LARGE_DATA if train else _ENFR_TEST_LARGE_DATA
    return datasets


@registry.register_problem
class TranslateEnfrWmtCharacters(TranslateEnfrWmtSmallCharacters):

  @property
  def use_small_dataset(self):
    return False


@registry.register_problem
class TranslateEnfrWmtMulti64k(TranslateEnfrWmtSmall32k):
  """Translation with muli-lingual vocabulary."""

  @property
  def use_small_dataset(self):
    return False

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelDeEnFrRoWiki64k()


@registry.register_problem
class TranslateEnfrWmtMulti64kPacked1k(TranslateEnfrWmtMulti64k):
  """Translation with muli-lingual vocabulary."""

  @property
  def packed_length(self):
    return 1024

  @property
  def num_training_examples(self):
    return 1760600

  @property
  def inputs_prefix(self):
    return "translate English French "

  @property
  def targets_prefix(self):
    return "translate French English "


@registry.register_problem
class TranslateLocalData(translate.TranslateProblem):

  @property
  def source_vocab_name(self):
    raise ValueError
    return "%s.input" % self.vocab_filename

  @property
  def target_vocab_name(self):
    raise ValueError
    return "%s.target" % self.vocab_filename

  @property
  def vocab_type(self):
    return VocabType.TOKEN

  @property
  def additional_training_datasets(self):
    # will be handled directly in generate_sample
    return []

  def source_data_files(self, dataset_split):
      return ['asd']

  def generate_samples(
      self,
      data_dir,
      tmp_dir,
      dataset_split,
      custom_iterator=text_problems.text2text_txt_iterator):
    tag = "dev"
    datatypes_to_clean = None
    if dataset_split == problem.DatasetSplit.TRAIN:
      tag = "train"
      datatypes_to_clean = self.datatypes_to_clean
    data_path = extract_data(
        tmp_dir, tag, "%s-compiled-%s" % (self.name, tag),
        datatypes_to_clean=datatypes_to_clean)
    return custom_iterator(data_path + ".lang1", data_path + ".lang2")

  def get_vocab(self, data_dir, is_target=False):
    vocab_filename = os.path.join(data_dir,
                                  self.target_vocab_name if is_target else self.source_vocab_name)
    if not tf.gfile.Exists(vocab_filename):
      raise ValueError("Vocab %s not found" % vocab_filename)
    return text_encoder.TokenTextEncoder(vocab_filename, replace_oov="UNK")

  def feature_encoders(self, data_dir):
    target_encoder = self.get_or_create_vocab(data_dir, None, force_get=True, target=True)
    encoders = {"targets": target_encoder}
    if self.has_inputs:
      encoder = self.get_or_create_vocab(data_dir, None, force_get=True, target=False)
      encoders["inputs"] = encoder
    return encoders

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    if dataset_split == problem.DatasetSplit.TRAIN:
      mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
    elif dataset_split == problem.DatasetSplit.EVAL:
      mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)

    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir, target=False)
    target_decoder = self.get_or_create_vocab(data_dir, tmp_dir, target=True)
    return text2text_generate_encoded(generator, encoder, target_decoder,
                                      has_inputs=self.has_inputs,
                                      inputs_prefix=self.inputs_prefix,
                                      targets_prefix=self.targets_prefix)

  def get_or_create_vocab(self, data_dir, tmp_dir, target, force_get=False):
    if FLAGS.vocab_type == 'subword':
      if force_get:
        vocab_filepath = os.path.join(data_dir, self.vocab_filename)
        encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
      else:
        other_problem = self.use_vocab_from_other_problem
        if other_problem:
          return other_problem.get_or_create_vocab(data_dir, tmp_dir, force_get)
        encoder = generator_utils.get_or_generate_vocab_inner(
            data_dir, self.vocab_filename, self.approx_vocab_size,
            self.generate_text_for_vocab(data_dir, tmp_dir),
            max_subtoken_length=self.max_subtoken_length,
            reserved_tokens=(
                text_encoder.RESERVED_TOKENS + self.additional_reserved_tokens))
    elif FLAGS.vocab_type == 'token':
      vocab_filename = os.path.join(data_dir, 'vocab.{}'.format('target' if target else 'input'))
      if not os.path.exists(vocab_filename):
        inputs, targets = get_input_target_names('train')
        if target:
          to_parse = targets
        else:
          to_parse = inputs
        with open(to_parse, 'r') as in_stream:

          max_vs = FLAGS.max_target_vocab_size if target else FLAGS.max_input_vocab_size
          if max_vs < 0:
            max_vs = None

          vocab = Vocab(in_stream, insert_additional_symbols={
            text_encoder.EOS: text_encoder.EOS_ID}, max_vocab_size=max_vs)
          tf.logging.info("created vocab for {} with {} entries".format(
            'target' if target else 'input', vocab.size()))
          assert vocab.get_tokens()[:3] == ['<pad>', '<EOS>', '<oov>']
          with open(vocab_filename, 'w') as out_stream:
            out_stream.write('\n'.join(vocab.reverse_vocab))
      encoder = OOVTokenTextEncoder(vocab_filename,
                                              replace_oov=self.oov_token)
    else:
      raise ValueError(
          "Unrecognized VocabType: %s" % str(FLAGS.vocab_type))
    return encoder

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    lang1_filepath, lang2_filepath = get_input_target_names('train')
    # note this is a shared vocab - so we go over both train files

    def generator():
      for file_path in [lang1_filepath, lang2_filepath]:
        with tf.gfile.GFile(file_path, mode="r") as file_stream:
          for line in file_stream:
            yield line

    return generator()

def extract_data(tmp_dir, data_split, filename, datatypes_to_clean=None):
  datatypes_to_clean = datatypes_to_clean or []
  lang1_out_fname = filename + ".lang1"
  lang2_out_fname = filename + ".lang2"
  if tf.gfile.Exists(lang1_out_fname) and tf.gfile.Exists(lang2_out_fname):
    tf.logging.info("Skipping compile data, found files:\n%s\n%s", lang1_out_fname,
                    lang2_out_fname)
    return filename

  lang1_filepath, lang2_filepath = get_input_target_names(data_split)
  tf.logging.info('input {} file: {}'.format(data_split, lang1_filepath))
  tf.logging.info('target {} file: {}'.format(data_split, lang2_filepath))

  with tf.gfile.GFile(lang1_out_fname, mode="w") as lang1_resfile:
    with tf.gfile.GFile(lang2_out_fname, mode="w") as lang2_resfile:
      for example in text_problems.text2text_txt_iterator(
          lang1_filepath, lang2_filepath):
        line1res = _preprocess_sgm(example["inputs"], False)
        line2res = _preprocess_sgm(example["targets"], False)
        clean_pairs = [(line1res, line2res)]
        if "txt" in datatypes_to_clean:
          clean_pairs = cleaner_en_xx.clean_en_xx_pairs(clean_pairs)
        for line1res, line2res in clean_pairs:
          if line1res and line2res:
            lang1_resfile.write(line1res)
            lang1_resfile.write("\n")
            lang2_resfile.write(line2res)
            lang2_resfile.write("\n")
  return filename


def get_input_target_names(data_split):
  lang_file_path = FLAGS.parsing_path
  lang1_filepath = lang_file_path + '/' + data_split + '.lang1'
  lang2_filepath = lang_file_path + '/' + data_split + '.lang2'
  return lang1_filepath, lang2_filepath


@registry.register_problem
class TranslateLocalDataMt2(TranslateLocalData):
  pass


@registry.register_problem
class LanguagemodelEnFrWiki32k(LanguagemodelEnWiki32k):
  """A language model on untokenized Wikipedia, 4 languages together."""

  train_names_ids = [_FR_TRAIN_NAME_ID, _EN_TRAIN_NAME_ID]
  eval_names_ids = [_FR_EVAL_NAME_ID, _EN_EVAL_NAME_ID]
  test_names_ids = [_FR_TEST_NAME_ID, _EN_TEST_NAME_ID]

  @property
  def approx_vocab_size(self):
    return 32000

  @property
  def max_samples_for_vocab(self):
    return 256000  # Samples are intertwined, take more to cover 4 languages.

@registry.register_problem
class LanguagemodelMultiLocalData(multi_problem.MultiProblem):
  """Wiki multi-lingual LM and multiple translations."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelMultiLocalData, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(LanguagemodelEnFrWiki32k())
    self.task_list.append(TranslateLocalData())
    self.task_list.append(TranslateLocalDataMt2())
    # self.task_list.append(translate_ende.TranslateEndeWmtMulti64k(
    #     was_reversed=True))
    # self.task_list.append(TranslateEnfrWmtMulti64k(
    #     was_reversed=True))
    # self.task_list.append(translate_enro.TranslateEnroWmtMultiTiny64k(
    #     was_reversed=True))
    # self.task_list.append(
    #     cnn_dailymail.SummarizeCnnDailymailWikiLMMultiVocab64k())
    # self.task_list.append(multinli.MultiNLIWikiLMMultiVocab64k())
    # self.task_list.append(squad.SquadConcatMulti64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD

@registry.register_problem
class LanguagemodelLocalData(text_problems.Text2SelfProblem):
  """A language model on the untokenized wikipedia corpus, English."""

  train_names_ids = [_EN_TRAIN_NAME_ID]
  eval_names_ids = [_EN_EVAL_NAME_ID]
  test_names_ids = [_EN_TEST_NAME_ID]

  @property
  def approx_vocab_size(self):
    return 32000

  @property
  def max_samples_for_vocab(self):
    return 128000

  @property
  def combine_characters_threshold(self):
    """Threshold for upto how many characters to combine in examples."""
    return 512*8  # So we should have 512 tokens on average, maybe more.

  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 100,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  def generate_samples(
      self,
      data_dir,
      tmp_dir,
      dataset_split,
      custom_iterator=text_problems.text2text_txt_iterator):
    tag = "dev"
    datatypes_to_clean = None
    if dataset_split == problem.DatasetSplit.TRAIN:
      tag = "train"
      # datatypes_to_clean = self.datatypes_to_clean
    data_path = extract_data(
        tmp_dir, tag, "%s-compiled-%s" % (self.name, tag),
        datatypes_to_clean=datatypes_to_clean)
    return custom_iterator(data_path + ".lang1", data_path + ".lang2")



@registry.register_problem
class LanguagemodelMultiLocalData2(multi_problem.MultiProblem):
  """Wiki multi-lingual LM and multiple translations."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(LanguagemodelMultiLocalData2, self).__init__(
        was_reversed, was_copy)
    self.task_list.append(LanguagemodelLocalData())
    self.task_list.append(TranslateLocalData())
    self.task_list.append(TranslateLocalDataMt2())
    # self.task_list.append(translate_ende.TranslateEndeWmtMulti64k(
    #     was_reversed=True))
    # self.task_list.append(TranslateEnfrWmtMulti64k(
    #     was_reversed=True))
    # self.task_list.append(translate_enro.TranslateEnroWmtMultiTiny64k(
    #     was_reversed=True))
    # self.task_list.append(
    #     cnn_dailymail.SummarizeCnnDailymailWikiLMMultiVocab64k())
    # self.task_list.append(multinli.MultiNLIWikiLMMultiVocab64k())
    # self.task_list.append(squad.SquadConcatMulti64k())

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD


@registry.register_hparams
def transformer_mt_local_data():
  """Hparams for transformer on LM pretraining on TPU, large model."""
  hparams = transformer_tall_pretrain_lm_tpu_adafactor()
  #hparams = transformer_tiny()
  hparams.hidden_size = 1024
  hparams.num_heads = 16
  hparams.filter_size = 32768  # max fitting in 16G memory is 49152, batch 2
  hparams.batch_size = 4
  hparams.multiproblem_mixing_schedule = "constant"
  hparams.multiproblem_per_task_threshold = "320,80,80"
  return hparams