"""
Encode images and assign the detected object labels.
NOTE: Run in tensorflow unsupervised image captioning directory.

Download models in `~/workspace/ckpt`:
    https://github.com/tensorflow/models/tree/master/research/slim
    wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
    wget http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz

NOTE: ResNetV2 models used Inception preprocess.
      https://github.com/tensorflow/models/tree/master/research/slim    (see below download link)
      https://github.com/tensorflow/models/issues/2217

Sample command:
    python -u process_store_image.py --image_path ~/dataset/mscoco/all_image --model inceptionv4 --model_ckpt ~/workspace/ckpt/inception_v4.ckpt --batch_size 64 --img_out_dir ~/mscoco_image_features
    python -u process_store_image.py --image_path ~/dataset/mscoco/all_image --model resnet101v2 --model_ckpt ~/workspace/ckpt/resnet_v2_101.ckpt --batch_size 64 --img_out_dir ~/mscoco_image_features
"""
import os, sys
import pickle as pkl

import h5py
import json
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import TF_MODELS_PATH
sys.path.append(TF_MODELS_PATH + '/research/im2txt/im2txt')
sys.path.append(TF_MODELS_PATH + '/research/slim')

from nets import inception_v4
from nets import resnet_v2

from absl import app
from absl import flags

from tqdm import tqdm
from tqdm import trange
from PIL import Image


flags.DEFINE_string('image_path', None, 'Path to all coco images.')
flags.DEFINE_string('model', None, 'model name in {inceptionv4, resnet101v2}')
flags.DEFINE_string('model_ckpt', None, 'checkpoint path to {inceptionv4, resnet101v2}')
flags.DEFINE_integer('batch_size', None, 'Batch size of image encoding')
flags.DEFINE_string('img_out_dir', None, 'Output directory of image features')

FLAGS = flags.FLAGS


def get_objects(split):
    FILE_NAME = 'data/img_obj_%s.json' % split
    if os.path.exists(FILE_NAME):
        print(FILE_NAME, "already exists")
    else:
        print("preparing", FILE_NAME, "...")
        obj_dict = {}
        with open('data/coco_%s.txt' % split, 'r') as f:
            filename = list(f)
            filename = [i.strip() for i in filename]
        with open('data/all_ids.pkl', 'rb') as f:
            all_ids = pkl.load(f)
        i2w = {}
        with open('data/word_counts.txt', 'r') as f:
            for line in f:
                word, freq = line.rstrip().split()
                i2w[len(i2w)] = word
        with h5py.File('data/object.hdf5', 'r') as f:
            for i in tqdm(filename):
                # NOTE: Objects below have their own score more than 0.3
                name = os.path.splitext(i)[0]
                detection_classes = f[name + '/detection_classes'][:].astype(np.int32)
                detection_scores = f[name + '/detection_scores'][:]
                detection_classes, ind = np.unique(detection_classes, return_index=True)
                detection_scores = detection_scores[ind]
                detection_classes = [all_ids[j] for j in detection_classes]
                detection_classes_word = [i2w[wid] for wid in detection_classes]
                obj_dict[i] = detection_classes_word
        with open(FILE_NAME, "w", encoding="utf-8") as outfile:
            json.dump(obj_dict, outfile, indent=4)
        print("dumped", FILE_NAME)


def read_image(im_list):
    """Reads an image.
    In the evaluation, tensorflow inception takes images whose values are normalized in [-1, 1].
    This normalization is different from that of train [-2, 0], but code above and the original unsupervised captioning model use this.
    
    https://github.com/tensorflow/models/blob/d1c48afcd0ca503bf8a320cfe862eed04217c68d/research/slim/preprocessing/inception_preprocessing.py#L280-L282
    https://github.com/tensorflow/models/issues/3346
    https://stackoverflow.com/questions/39582703/using-pre-trained-inception-resnet-v2-with-tensorflow/39597537#39597537
    https://github.com/fengyang0317/unsupervised_captioning/blob/master/caption_infer.py
    """
    im_batch = list()
    for im in im_list:
        filename = os.path.join(FLAGS.image_path, im)
        image = Image.open(filename).convert("RGB")
        image = image.resize((346, 346), Image.BILINEAR)   # tf.resize take bilinear as default
        image = np.array(image, dtype=np.float32)[23:-24, 23:-24, :]    # ndarray: (299, 299, 3)
        image = image / 255     # normalize to [0, 1], which is implicitly applied in `tf.image.convert_image_dtype`
        image = image * 2 - 1   # this convert image in the range of [-1, 1]
        im_batch.append(image)
    im_batch = np.stack(im_batch, axis=0)   # (batch, 299, 299, 3)
    return im_batch


def output_features(image_batch):
    if FLAGS.model == 'inceptionv4':
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            net, _ = inception_v4.inception_v4(image_batch, None, is_training=False)
            net = tf.squeeze(net, [1, 2])
    elif FLAGS.model == 'resnet101v2':
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            net, _ = resnet_v2.resnet_v2_101(image_batch, None, is_training=False, global_pool=True)
            net = tf.squeeze(net, [1, 2])
    else:
        raise KeyError('{} is not supported'.format(FLAGS.model))
    return net


class Encoder:

    def __init__(self):
        H = 299
        W = 299
        C = 3
        image_batch = tf.placeholder(tf.float32, [FLAGS.batch_size, H, W, C])
        output_features_op = output_features(image_batch)

        self.image_batch = image_batch
        self.output_features = output_features_op
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=config)

        # NOTE: saver name must be the `scope` of the model class
        #       https://github.com/tensorflow/models/blob/d1c48afcd0ca503bf8a320cfe862eed04217c68d/research/slim/nets/inception_v4.py   (l-260)
        #       https://github.com/tensorflow/models/blob/d1c48afcd0ca503bf8a320cfe862eed04217c68d/research/slim/nets/resnet_v2.py  (l-281)
        if FLAGS.model == 'inceptionv4':
            model_saver = tf.train.Saver(tf.global_variables('InceptionV4'))
        elif FLAGS.model == 'resnet101v2':
            model_saver = tf.train.Saver(tf.global_variables('resnet_v2_101'))
        else:
            raise KeyError('{} is not supported'.format(FLAGS.model))
        model_saver.restore(self.sess, FLAGS.model_ckpt)
    
    def get_features(self, split):
        FILE_NAME = os.path.join(FLAGS.img_out_dir , 'img_{}_{}.hdf5'.format(FLAGS.model, split))
        if os.path.exists(FILE_NAME):
            print(FILE_NAME, "already exists")
        else:
            print("preparing", FILE_NAME, "...")
            with open('data/coco_%s.txt' % split, 'r') as f:
                filename = list(f)
                filename = [i.strip() for i in filename]
            with h5py.File(FILE_NAME, 'w') as f:
                for i in trange(len(filename[::FLAGS.batch_size])):
                    image_list = filename[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                    image_len = len(image_list)
                    if image_len < FLAGS.batch_size:
                        image_list += [image_list[0]] * (FLAGS.batch_size - image_len)
                    image_batch = read_image(image_list)    # ndarray: (batch, H, W, 3)
                    net = self.sess.run(
                        self.output_features, feed_dict={self.image_batch: image_batch}
                    )   # (batch, dim)
                    for img, feat in zip(image_list[:image_len], net[:image_len]):
                        feat = np.expand_dims(feat, 0)
                        f.create_dataset(img, data=feat)
            print("dumped", FILE_NAME)


def main(_):
    encoder = Encoder()
    for i in ['val', 'test', 'train']:
        get_objects(i)
        encoder.get_features(i)


if __name__ == '__main__':
    app.run(main)
