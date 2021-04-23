"""Evaluates the performance of all the checkpoints on validation set.
   Sample command:
     python -u eval_all_fast.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt --job_dir saving_self_imcap --device 0 --threads 12
"""
import glob
import json
import multiprocessing
import os

from absl import app
from absl import flags

flags.DEFINE_integer('threads', 1, 'num of threads')

flags.DEFINE_string('img_dir', '~/mscoco_image_features', 'image features dir')

flags.DEFINE_string('img_file', 'img_inceptionv4_val.hdf5', 'image features file: [img_inceptionv4_val.hdf5, img_resnet101v2_val.hdf5]')

flags.DEFINE_string('gts_file', 'val_test_dict.json', 'ground-truth caption file: [val_test_dict.json, val_test_dict_v4.json]')

flags.DEFINE_string('device', '0', 'device')

from caption_infer_fast import Infer

import h5py
from speaksee import evaluation

FLAGS = flags.FLAGS


def initializer():
  """Decides which GPU is assigned to a worker.

  If your GPU memory is large enough, you may put several workers in one GPU.
  """
  global tf, no_gpu
  no_gpu = False
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device
  import tensorflow as tf

  global feat_hdf5, gts_dict
  feat_hdf5 = h5py.File(os.path.join(FLAGS.img_dir, FLAGS.img_file), "r")
  with open(os.path.join('./data/', FLAGS.gts_file)) as f:
    gts_dict = json.load(f)
    gts_dict = gts_dict['val']


def metric_eval(gen, gts):
  gen = evaluation.PTBTokenizer.tokenize(gen) # {0:["gen0"], ...}
  gts = evaluation.PTBTokenizer.tokenize(gts) # {0:["gt0-0", ..., "gt0-4"], ...}
  score_dict = dict()
  bleu, _ = evaluation.Bleu(n=4).compute_score(gts, gen)
  method = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4']
  for metric, score in zip(method, bleu):
    score_dict[metric] = score
  cider, _ = evaluation.Cider().compute_score(gts, gen)
  score_dict['CIDEr'] = cider
  # meteor, _ = evaluation.Meteor().compute_score(gts, gen)
  # score_dict['METEOR'] = meteor
  # rouge, _ = evaluation.Rouge().compute_score(gts, gen)
  # score_dict['ROUGE_L'] = rouge
  # spice, _ = evaluation.Spice().compute_score(gts, gen)
  # score_dict['SPICE'] = spice
  return score_dict   # {'metric':score, ...}


def run(inp):
  if no_gpu:
    print('No GPU')
    return
  
  print('Start evaluating:', '%s/model.ckpt-%s' % (FLAGS.job_dir, inp))
  gen = []
  gts = []
  with tf.Graph().as_default():
    infer = Infer(job_dir='%s/model.ckpt-%s' % (FLAGS.job_dir, inp))
    with open('data/coco_val.txt', 'r') as g:
      for name in g:
        name = name.strip()
        feat = feat_hdf5[name][()]    # (1, dim)
        sentences = infer.infer(feat)
        gen.append(sentences[0][0]) # ["seq", ...]
        gts.append(gts_dict[name])  # [["gt1", ..., "gt5"], ...]
  assert len(gen) == len(gts) == 5000

  score_dict = metric_eval(gen, gts)
  score_dict['ckpt'] = inp
  print('Finished evaluating:', '%s/model.ckpt-%s' % (FLAGS.job_dir, inp))
  return score_dict


def get_best(dict_list, metric):
  assert metric in {'CIDEr', 'BLEU_3-4'}
  best_score = -1
  best_ckpt = -1
  for dict_item in dict_list:
    if metric == 'CIDEr':
      curr_score = dict_item['CIDEr']
    else:
      curr_score = dict_item['BLEU_3'] + dict_item['BLEU_4']
    if curr_score > best_score:
      best_score = curr_score
      best_ckpt = dict_item['ckpt']
  print('Best ckpt in {}: {}'.format(metric, best_ckpt))
  print('Best score in {}: {}'.format(metric, best_score))


def main(_):
  results = glob.glob(FLAGS.job_dir + '/model.ckpt-*')
  results = [os.path.splitext(i)[0] for i in results]
  results = set(results)
  gs_list = [i.split('-')[-1] for i in results]
  print('Checkpoints to evaluate:', gs_list)

  pool = multiprocessing.Pool(FLAGS.threads, initializer)
  ret = pool.map(run, gs_list)  # [{score_dict}, ...]
  pool.close()
  pool.join()
  if not ret or ret[0] is None:
    print('No Output')
    return

  print(ret)
  get_best(ret, 'CIDEr')
  get_best(ret, 'BLEU_3-4')
  print('\nFinished all evaluation')


if __name__ == '__main__':
  app.run(main)