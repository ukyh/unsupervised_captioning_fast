"""Generate pseudo captions.

python -u initialization/gen_self_caption.py --image_path ~/dataset/mscoco/all_images --selfcap selfcap_ss4_mi2.json
"""

import multiprocessing
import os
from functools import partial

import h5py
import json
import pickle as pkl
import numpy as np

from absl import app
from absl import flags

from misc_fn import _bytes_feature
from misc_fn import _float_feature_list
from misc_fn import _int64_feature_list

flags.DEFINE_integer('num_proc', 1, 'number of processes')

flags.DEFINE_integer('num_gpus', 1, 'number of gpus')

flags.DEFINE_string('selfcap', 'selfcap_ss4_mi2.json', 'self caption path')

flags.DEFINE_string('image_path', None, 'Path to all coco images.')

flags.DEFINE_bool('skip_imgen', False, 'skip generation of image_selftrain.tfrec')

from sentence_infer import Infer

FLAGS = flags.FLAGS


fname_list = list()
def load_orderd_fname():
  global fname_list
  with open('data/coco_train.txt', 'r', encoding='utf-8') as f:
    filename = list(f)
    fname_list = [i.strip() for i in filename]
  print('#### Loaded ordered image files')


def load_selfcaps(fname):
  self_caps = dict()
  with open('data/' + fname, 'r', encoding='utf-8') as f:
    _self_caps = json.load(f) # [{'image_id':str, 'caption':str}, ....]
    for items in _self_caps:
      self_caps[items['image_id']] = items['caption']
  return self_caps


def initializer():
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  global infer
  infer = Infer()
  global self_caps
  self_caps = load_selfcaps(FLAGS.selfcap)


def iter_iname():
    for iname in fname_list:
      yield iname


def run(iname):
  tf = infer.tf
  if iname in self_caps:
    sentence = self_caps[iname].split()
    sentence = [infer.vocab.word_to_id(i) for i in sentence]
    context = tf.train.Features()
    feature_lists = tf.train.FeatureLists(feature_list={
    'sentence': _int64_feature_list(sentence)
    })
    sequence_example = tf.train.SequenceExample(
    context=context, feature_lists=feature_lists)
    return sequence_example.SerializeToString()
  else:
    sentence = [-1]
    context = tf.train.Features()
    feature_lists = tf.train.FeatureLists(feature_list={
    'sentence': _int64_feature_list(sentence)
    })
    sequence_example = tf.train.SequenceExample(
    context=context, feature_lists=feature_lists)
    return sequence_example.SerializeToString()


def image_generator(tf):
  with open('data/coco_train.txt', 'rb') as f:
    filename = list(f)
    filename = [i.strip() for i in filename]
  with open('data/all_ids.pkl', 'rb') as f:
    all_ids = pkl.load(f)
  with h5py.File('data/object.hdf5', 'r') as f:
    for i in filename:
      name = os.path.splitext(i.decode())[0]
      detection_classes = f[name + '/detection_classes'][:].astype(np.int32)
      detection_scores = f[name + '/detection_scores'][:]
      detection_classes, ind = np.unique(detection_classes, return_index=True)
      detection_scores = detection_scores[ind]
      detection_classes = [all_ids[j] for j in detection_classes]
      image_path = FLAGS.image_path + '/' + i.decode()
      with tf.gfile.FastGFile(image_path, 'rb') as g:
        image = g.read()
      context = tf.train.Features(feature={
        'image/name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[i])),
        'image/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
      })
      feature_lists = tf.train.FeatureLists(feature_list={
        'classes': _int64_feature_list(detection_classes),
        'scores': _float_feature_list(detection_scores)
      })
      sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)
      yield sequence_example.SerializeToString()


def gen_tfrec(tf):
  ds = tf.data.Dataset.from_generator(partial(image_generator, tf=tf),
                                      output_types=tf.string, output_shapes=())
  tfrec = tf.data.experimental.TFRecordWriter('data/image_selftrain.tfrec')
  tfrec.write(ds)


def main(_):
  pool = multiprocessing.Pool(FLAGS.num_proc, initializer=initializer)
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  import tensorflow as tf
  tf.enable_eager_execution()
  load_orderd_fname()
  OUT_FILE = FLAGS.selfcap.rstrip('json') + 'tfrec'
  with tf.python_io.TFRecordWriter('data/' + OUT_FILE) as writer:
    for i in pool.imap(run, iter_iname()):
      writer.write(i)
  print("#### Made selfcap tfrec")
  if not FLAGS.skip_imgen:
    gen_tfrec(tf)
  print("#### Finished processes")


if __name__ == '__main__':
  app.run(main)