# Unsupervised Image Captioning Fast


This is a slightly modified version of [unsupervised_captioning](https://github.com/fengyang0317/unsupervised_captioning) (many thanks to [fengyang0317](https://github.com/fengyang0317)). We modified its evaluation to be faster.   
Our work, [RemovingSpuriousAlignment](https://github.com/ukyh/RemovingSpuriousAlignment) uses these codes to:
* **Preprocess** and store image features
* **Combine** the methods of [unsupervised_captioning](https://github.com/fengyang0317/unsupervised_captioning) with our methods


## Requirements
1. Run the following commands to setup.
```bash
mkdir ~/workspace
cd ~/workspace
git clone https://github.com/tensorflow/models.git tf_models
cd tf_models
git checkout 403014db9f12c0228529db2c8b292efcced5133a
cd ..
git clone https://github.com/tylin/coco-caption.git
touch tf_models/research/im2txt/im2txt/__init__.py
touch tf_models/research/im2txt/im2txt/data/__init__.py
touch tf_models/research/im2txt/im2txt/inference_utils/__init__.py
wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
wget http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz
mkdir ckpt
tar zxvf inception_v4_2016_09_09.tar.gz -C ckpt
tar zxvf resnet_v2_101_2017_04_14.tar.gz -C ckpt
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz
tar -xzvf faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz -C tf_models/research/object_detection
git clone https://github.com/ukyh/unsupervised_captioning_fast.git
cd unsupervised_captioning_fast
pip install -r requirements.txt
mkdir saving_self
mkdir saving_self_imcap
mkdir sen_gan
mkdir obj2sen
export PYTHONPATH=$PYTHONPATH:`pwd`
```
**Tips**  
If you have trouble in importing `tensorflow==1.13.1`, try below (see [this discussion](https://github.com/tensorflow/tensorflow/issues/26182) for the details).
```bash
conda install -c anaconda cudatoolkit==10.0.130 cudnn==7.6.5
```

2. Download images from [MS COCO](http://cocodataset.org/) to `~/dataset/mscoco/all_images`.
```bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip -j train2014.zip -d ~/dataset/mscoco/all_images
unzip -j val2014.zip -d ~/dataset/mscoco/all_images
```

3. Download [ucap_fast_data.tar.gz](https://drive.google.com/file/d/1A2tUhQDqbdHhQzescsMvIxXJZThLR2h_/view?usp=sharing) and unpack it in `unsupervised_captioning_fast/data`.

**For combined method**  

4. Download `model.ckpt-30000.*` files from [here](https://github.com/fengyang0317/unsupervised_captioning) and put them into  `unsupervised_captioning_fast/sen_gan`.

5. Download `image_train.tfrec` and `sentence.tfrec` to `unsupervised_captioning_fast/data`. To download these file, please refer to [this discussion](https://github.com/fengyang0317/unsupervised_captioning/issues/36) (currently, `image_train.tfrec` is only available in the Baidu link).


**Acknowledgement**
* `all_ids.pkl` and `object.hdf5` are provided by [unsupervised_captioning](https://github.com/fengyang0317/unsupervised_captioning)
* `oid_v4_label_map.pbtxt` is provided by [tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/data/oid_v4_label_map.pbtxt)


## Preprocess
```bash
cd unsupervised_captioning_fast
export PYTHONPATH=$PYTHONPATH:`pwd`
```

For Feng et al. (2019) setting, run the following command.
```bash
python -u preprocessing/process_store_image.py --image_path ~/dataset/mscoco/all_images --model inceptionv4 --model_ckpt ~/workspace/ckpt/inception_v4.ckpt --batch_size 64 --img_out_dir ~/mscoco_image_features
```

For Laina et al. (2019) setting, run the following commands.
```bash
python -u preprocessing/process_store_image.py --image_path ~/dataset/mscoco/all_images --model resnet101v2 --model_ckpt ~/workspace/ckpt/resnet_v2_101.ckpt --batch_size 64 --img_out_dir ~/mscoco_image_features
python -u preprocessing/detect_objects_v4.py --image_path ~/dataset/mscoco/all_images --num_proc 4 --num_gpus 4
python -u preprocessing/process_oid_v4.py ./data
```


## Combine
This step requires a `.json` file of generated captions for MS COCO training images.
To make the file of the generated captions, please refer to `Preprocess to Combine` instruction of [RemovingSpuriousAlignment](https://github.com/ukyh/RemovingSpuriousAlignment). After creating the file, copy it to `unsupervised_captioning_fast/data`.  

```bash
cd unsupervised_captioning_fast
export PYTHONPATH=$PYTHONPATH:`pwd`
```

1. Create a `.tfrec` file by the generated captions of our work. The name of the input caption file is supposed to be `selfcap_ss4_mi2.json` in this example.
```bash
python -u initialization/gen_self_caption.py --image_path ~/dataset/mscoco/all_images --selfcap selfcap_ss4_mi2.json
```

2. Initialize a caption generator with the generated captions.
```bash
python -u initialization/im_caption_self.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt --batch_size 512 --multi_gpu --selfcap selfcap_ss4_mi2.tfrec --job_dir saving_self_imcap
```

3. Choose the best generator by the BLEU_3-4 score of the validation set.
```bash
python -u eval_all_fast.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt --device 0 --threads 12 --job_dir saving_self_imcap 
```

4. Apply the methods of [unsupervised_captioning](https://github.com/fengyang0317/unsupervised_captioning) with the best initialized generator. The best initialized generator is supposed to be `saving_self_imcap/model.ckpt-4000` in this example.
```bash
python -u im_caption_full.py \
  --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt \
  --imcap_ckpt saving_self_imcap/model.ckpt-4000 \
  --sae_ckpt sen_gan/model.ckpt-30000 \
  --multi_gpu \
  --batch_size 512 \
  --save_checkpoint_steps 100 \
  --gen_lr 0.00001 \
  --dis_lr 0.00000001 \
  --max_steps 1000 \
  --job_dir saving_self
```

5. Choose the best generator by the BLEU_3-4 score of the validation set.
```bash
python -u eval_all_fast.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt --device 0 --threads 12 --job_dir saving_self
```

6. Test the best generator. The best generator is supposed to be `saving_self/model.ckpt-500` in this example.
```bash
python -u test_model_fast.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt --job_dir saving_self/model.ckpt-500
```


## References
* Yang Feng, Lin Ma, Wei Liu, and Jiebo Luo. 2019. Unsupervised image captioning. In _CVPR_.
* Iro Laina, Christian Rupprecht, and Nassir Navab. 2019. Towards unsupervised image captioning with shared multimodal embeddings. In _ICCV_.


---

## Inherited

### Introduction
Most image captioning models are trained using paired image-sentence data, which
are expensive to collect. We propose unsupervised image captioning to relax the 
reliance on paired data. For more details, please refer to our
[paper](https://arxiv.org/abs/1811.10787).

![alt text](http://cs.rochester.edu/u/yfeng23/cvpr19_captioning/framework.png 
"Framework")

### Citation

    @InProceedings{feng2019unsupervised,
      author = {Feng, Yang and Ma, Lin and Liu, Wei and Luo, Jiebo},
      title = {Unsupervised Image Captioning},
      booktitle = {CVPR},
      year = {2019}
    }

### Requirements
```
mkdir ~/workspace
cd ~/workspace
git clone https://github.com/tensorflow/models.git tf_models
git clone https://github.com/tylin/coco-caption.git
touch tf_models/research/im2txt/im2txt/__init__.py
touch tf_models/research/im2txt/im2txt/data/__init__.py
touch tf_models/research/im2txt/im2txt/inference_utils/__init__.py
wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
mkdir ckpt
tar zxvf inception_v4_2016_09_09.tar.gz -C ckpt
git clone https://github.com/fengyang0317/unsupervised_captioning.git
cd unsupervised_captioning
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:`pwd`
```

### Dataset (Optional. The files generated below can be found at [Gdrive][1]).
In case you do not have the access to Google, the files are also available at
[One Drive][2].
1. Crawl image descriptions. The descriptions used when conducting the
experiments in the paper are available at
[link](https://drive.google.com/file/d/1z8JwNxER-ORWoAmVKBqM7MyPozk6St4M).
You may download the descriptions from the link and extract the files to
data/coco.
    ```
    pip3 install absl-py
    python3 preprocessing/crawl_descriptions.py
    ```

2. Extract the descriptions. It seems that NLTK is changing constantly. So 
the number of the descriptions obtained may be different.
    ```
    python -c "import nltk; nltk.download('punkt')"
    python preprocessing/extract_descriptions.py
    ```

3. Preprocess the descriptions. You may need to change the vocab_size, start_id,
and end_id in config.py if you generate a new dictionary.
    ```
    python preprocessing/process_descriptions.py --word_counts_output_file \ 
      data/word_counts.txt --new_dict
    ```

4. Download the MSCOCO images from [link](http://cocodataset.org/) and put 
all the images into ~/dataset/mscoco/all_images.

5. Object detection for the training images. You need to first download the
detection model from [here][detection_model] and then extract the model under
tf_models/research/object_detection.
    ```
    python preprocessing/detect_objects.py --image_path\
      ~/dataset/mscoco/all_images --num_proc 2 --num_gpus 1
    ```

6. Generate tfrecord files for images.
    ```
    python preprocessing/process_images.py --image_path\
      ~/dataset/mscoco/all_images
    ```

### Training
7. Train the model without the intialization pipeline.
    ```
    python im_caption_full.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt\
      --multi_gpu --batch_size 512 --save_checkpoint_steps 1000\
      --gen_lr 0.001 --dis_lr 0.001
    ```

8. Evaluate the model. The last element in the b34.json file is the best
checkpoint.
    ```
    CUDA_VISIBLE_DEVICES='0,1' python eval_all.py\
      --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt\
      --data_dir ~/dataset/mscoco/all_images
    js-beautify saving/b34.json
    ```

9. Evaluate the model on test set. Suppose the best validation checkpoint
is 20000.
    ```
    python test_model.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt\
      --data_dir ~/dataset/mscoco/all_images --job_dir saving/model.ckpt-20000
    ```

### Initialization (Optional. The files can be found at [here][1]).

10. Train a object-to-sentence model, which is used to generate the
pseudo-captions.
    ```
    python initialization/obj2sen.py
    ```

11. Find the best obj2sen model.
    ```
    python initialization/eval_obj2sen.py --threads 8
    ```

12. Generate pseudo-captions. Suppose the best validation checkpoint is 35000.
    ```
    python initialization/gen_obj2sen_caption.py --num_proc 8\
      --job_dir obj2sen/model.ckpt-35000
    ```

13. Train a captioning using pseudo-pairs.
    ```
    python initialization/im_caption.py --o2s_ckpt obj2sen/model.ckpt-35000\
      --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt
    ```

14. Evaluate the model.
    ```
    CUDA_VISIBLE_DEVICES='0,1' python eval_all.py\
      --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt\
      --data_dir ~/dataset/mscoco/all_images --job_dir saving_imcap
    js-beautify saving_imcap/b34.json
    ```

15. Train sentence auto-encoder, which is used to initialize sentence GAN.
    ```
    python initialization/sentence_ae.py
    ```

16. Train sentence GAN.
    ```
    python initialization/sentence_gan.py
    ```

17. Train the full model with initialization. Suppose the best imcap validation
checkpoint is 18000.
    ```
    python im_caption_full.py --inc_ckpt ~/workspace/ckpt/inception_v4.ckpt\
      --imcap_ckpt saving_imcap/model.ckpt-18000\
      --sae_ckpt sen_gan/model.ckpt-30000 --multi_gpu --batch_size 512\
      --save_checkpoint_steps 1000 --gen_lr 0.001 --dis_lr 0.001
    ```

### Credits
Part of the code is from 
[coco-caption](https://github.com/tylin/coco-caption),
[im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt),
[tfgan](https://github.com/tensorflow/models/tree/master/research/gan),
[resnet](https://github.com/tensorflow/models/tree/master/official/resnet),
[Tensorflow Object Detection API](
https://github.com/tensorflow/models/tree/master/research/object_detection) and
[maskgan](https://github.com/tensorflow/models/tree/master/research/maskgan).

[Xinpeng](https://github.com/chenxinpeng) told me the idea of self-critic, which
is crucial to training.

[1]: https://drive.google.com/drive/folders/1ol8gLj6hYgluldvdj9XFKm16TCqOr7EE
[2]: https://uofr-my.sharepoint.com/:f:/g/personal/yfeng23_ur_rochester_edu/EgDosCuY5t9HmlBfFsVyxdAB4xGf6aTJ0DmQlYWASdjYsw?e=Rhc4nS
[detection_model]: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz