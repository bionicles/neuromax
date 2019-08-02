import json
import os
import tensorflow as tf
from progiter import ProgIter
from imageio import imread
import pandas as pd
import time
SHARDS_PER_DATASET = 100000
ITEMS_PER_SHARD = 16
DELETE_RECORDS = True
images_path = "./datasets/CLEVR_v1.0/images"
data_path = "./datasets/CLEVR_v1.0/questions/CLEVR_train_questions.json"
tfrecord_path = "datasets/tfrecords"
NUM_FILES = len(os.listdir(os.path.join(tfrecord_path, "clevr")))
PREVIOUS_IMAGE = ITEMS_PER_SHARD * NUM_FILES

answers = ['0','1','2','3','4','5','6','7','8','9','10','blue','brown','cube','cyan','cylinder','gray','green','large','metal','no','purple','red','rubber','small','sphere','yellow','yes']


def load_data(image_data):
    image_filename = image_data['image_filename']
    question = image_data['question']
    string_answers = image_data['answer'].split(",")
    index_answers = [answers.index(element) for element in string_answers]
    answer = [tf.convert_to_tensor([0 if x != i else 1 for x in range(28)]) for i in index_answers]
    #split = image_data['split']
    image_path = os.path.join(images_path, "train", image_filename)
    question_bytes = bytes(question, 'utf-8')
    answer_bytes = bytes(str(answer), 'utf-8')
    image_bytes = tf.io.serialize_tensor(imread(image_path)).numpy()  # <--- this makes image into bytes
    return make_example(image_bytes, question_bytes, answer_bytes)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_example(image_bytes, question_bytes, answer_bytes):
    feature = {'question': _bytes_feature(question_bytes), 'answer': _bytes_feature(answer_bytes), 'image': _bytes_feature(image_bytes)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def write_shards(tfrecord_path):
    global PREVIOUS_IMAGE
    problems = []
    print("loading clevr JSON file, please wait.")
    clevr_data = json.load(open(data_path))
    print("done loading clevr JSON file")
    print("converting data to dataframe")
    clevr_data = pd.DataFrame(clevr_data["questions"])
    clevr_data.set_index("image_filename", inplace=True)
    question = clevr_data.groupby("image_filename")['question'].apply(lambda x: ','.join(x.astype(str))).reset_index().drop("image_filename", axis=1)
    answer = clevr_data.groupby("image_filename")['answer'].apply(lambda x: ','.join(x.astype(str))).reset_index()
    clevr_data = pd.concat([question, answer], axis=1)
    del question, answer
    NUM_FILES = len(os.listdir(tfrecord_path))
    shard_number = NUM_FILES-1 if NUM_FILES > 0 else 0
    for case in ProgIter(range(clevr_data.shape[0]), verbose=1):
        try:
            writer.close()
        except Exception as e:
            print('writer.close() exception', e)
        shard_path = os.path.join(tfrecord_path, 'clevr')
        if not os.path.exists(shard_path):
            os.makedirs(shard_path)
        shard_path = os.path.join(shard_path, str(shard_number) + '.tfrecord')
        writer = tf.io.TFRecordWriter(shard_path, 'ZLIB')
        shard_number += 1
        if shard_number > SHARDS_PER_DATASET:
            break
        try:
            data = load_data(clevr_data.loc[case])
            if data:
                writer.write(data)
                print('wrote', case, 'to', shard_path)
            else:
                print('skipped writing',  case, 'to', shard_path)
        except Exception as e:
            print('failed on', shard_number,  case, shard_path)
            print(e)
            problems.append([shard_number,  case, e])
    print('problem children:')
    [print(problem) for problem in problems]
    print('done!')


write_shards(tfrecord_path)
