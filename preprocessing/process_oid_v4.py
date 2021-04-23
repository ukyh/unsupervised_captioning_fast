# Based on:
#   https://github.com/fengyang0317/unsupervised_captioning/blob/master/preprocessing/process_descriptions.py
#   https://github.com/fengyang0317/unsupervised_captioning/blob/master/preprocessing/process_images.py

import sys
import os
import json
import re
import numpy as np
from urllib import request
from urllib.request import urlopen

import h5py
from tqdm import tqdm


DATA_PATH = sys.argv[1]
tag = re.compile(r"<[^>]*?>")
hyphen = re.compile(r"-")


def get_plural(word):
    c = re.compile('Noun</span> <p> \(.*<i>plural</i> ([^\)]+)\)')
    req = request.Request('https://www.yourdictionary.com/' + word, headers={'User-Agent': 'Magic Browser'})
    f = urlopen(req)
    html = f.read()
    f.close()
    html = html.decode('utf-8', errors="ignore")
    plural_word = c.findall(html)
    if plural_word:
        plural_word = plural_word[0]    # ["plural"] -> "plural"
        plural_word = plural_word.lower()
        # remove tags and hyphen
        plural_word = tag.sub("", plural_word)
        plural_word = hyphen.sub("", plural_word)
        plural_word = plural_word.split()[0]
    elif 'Noun</span> <p> (<i>plural only)' in html:
        plural_word = word
    else:
        plural_word = word
        if word[-1] != 's':
            plural_word += 's'
    return plural_word


def get_open_image_categories():
    print('Prepareing OID converter ...')
    with open(os.path.join(DATA_PATH, "oid_v4_label_map.pbtxt"), encoding="utf-8", errors="ignore") as f:
        categories = {}
        cid = -1
        dname = ""
        for line in f:
            if line.startswith("}"):
                if cid != -1 and dname != "":
                    categories[cid] = dname
                else:
                    raise ValueError("Something wrong in category file: cid {}, name {}".format(cid, dname))
                cid = -1
                dname = ""
            elif "id:" in line:
                cid = int(line.split()[-1])
            elif "display_name:" in line:
                dname = line.split()[-1].strip('"') # extract the last word (same in the original code)
                dname = dname.lower().split()[-1]   # lower the case and take the last word
            else:
                pass
    category_name = list(set(categories.values()))
    category_name.sort()
    print('Prepareing plural dict ...')
    plural_file = os.path.join(DATA_PATH, 'plural_words_v4.json')
    if os.path.exists(plural_file):
        with open(plural_file, 'r') as f:
            plural_dict = json.load(f)
            plural_name = [plural_dict[i] for i in category_name]
    else:
        plural_name = []
        for i in tqdm(category_name):
            plural_name.append(get_plural(i))
        with open(plural_file, 'w') as f:
            json.dump(dict(zip(category_name, plural_name)), f, indent=4)
    return category_name, plural_name, categories


def get_img_obj(split, category_dict):
    FILE_NAME = os.path.join(DATA_PATH, 'img_obj_%s_v4.json' % split)
    if os.path.exists(FILE_NAME):
        print(FILE_NAME, "already exists")
    else:
        print("preparing", FILE_NAME, "...")
        obj_dict = {}
        with open(os.path.join(DATA_PATH, 'coco_%s.txt' % split), 'r') as f:
            filename = list(f)
            filename = [i.strip() for i in filename]
        with h5py.File(os.path.join(DATA_PATH, 'object_v4.hdf5'), 'r') as f:
            for i in tqdm(filename):
                # NOTE: objects with scores >= 0.3 are detected 
                name = os.path.splitext(i)[0]
                detection_classes = f[name + '/detection_classes'][:].astype(np.int32)
                detection_scores = f[name + '/detection_scores'][:]
                detection_classes, ind = np.unique(detection_classes, return_index=True)
                detection_scores = detection_scores[ind]
                detection_classes_word = [category_dict[wid] for wid in detection_classes]
                obj_dict[i] = detection_classes_word
        with open(FILE_NAME, "w", encoding="utf-8") as outfile:
            json.dump(obj_dict, outfile, indent=4)


category_list, plural_list, category_dict = get_open_image_categories()
for data_split in ['train', 'val', 'test']:
    get_img_obj(data_split, category_dict)


print('Finished preparing OID v4')
