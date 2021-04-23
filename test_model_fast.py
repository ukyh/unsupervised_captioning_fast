"""Evaluate the performance on test split.
   Sample command:
     python -u test_model_fast.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt --job_dir saving_self_imcap/model.ckpt-40000
     python -u test_model_fast.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt --job_dir saving_self/model.ckpt-500
"""
import json
import os

from absl import app
from absl import flags
from tqdm import tqdm

from caption_infer_fast import Infer

import h5py
from speaksee import evaluation

flags.DEFINE_bool('vis', False, 'visulaize')
flags.DEFINE_string('device', '0', 'device')

flags.DEFINE_string('img_dir', '~/mscoco_image_features', 'image features dir')
flags.DEFINE_string('img_file', 'img_inceptionv4_test.hdf5', 'image features file: [img_inceptionv4_test.hdf5, img_resnet101v2_test.hdf5]')
flags.DEFINE_string('gts_file', 'val_test_dict.json', 'ground-truth caption file: [val_test_dict.json, val_test_dict_v4.json]')

FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device


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
  meteor, _ = evaluation.Meteor().compute_score(gts, gen)
  score_dict['METEOR'] = meteor
  rouge, _ = evaluation.Rouge().compute_score(gts, gen)
  score_dict['ROUGE_L'] = rouge
  spice, _ = evaluation.Spice().compute_score(gts, gen)
  score_dict['SPICE'] = spice
  return score_dict   # {'metric':score, ...}


def main(_):
  infer = Infer()
  feat_hdf5 = h5py.File(os.path.join(os.path.expanduser(FLAGS.img_dir), FLAGS.img_file), "r")
  with open(os.path.join('./data/', FLAGS.gts_file)) as f:
    gts_dict = json.load(f)
    gts_dict = gts_dict['test']

  with open('data/coco_test.txt', 'r') as g:
    ret = []
    gen = []
    gts = []
    for name in tqdm(g, total=5000):
      name = name.strip()
      feat = feat_hdf5[name][()]  # (1, dim)
      sentences = infer.infer(feat)
      cur = {}
      cur['image_id'] = name
      cur['caption'] = sentences[0][0]
      ret.append(cur)
      gen.append(sentences[0][0]) # ["seq", ...]
      gts.append(gts_dict[name])  # [["gt1", ..., "gt5"], ...]
    assert len(gen) == len(gts) == 5000

  if os.path.isdir(FLAGS.job_dir):
    out_dir = FLAGS.job_dir
  else:
    out_dir = os.path.split(FLAGS.job_dir)[0]
  out = out_dir + '/test.json'
  with open(out, 'w') as g:
    json.dump(ret, g, indent=4)

  score_dict = metric_eval(gen, gts)
  print('Evaluate on:', FLAGS.job_dir)
  print('Test\t{}'.format(' '.join([':'.join([k, str(v)]) for k, v in score_dict.items()])))
  print('\nFinished evaluation')


if __name__ == '__main__':
  app.run(main)