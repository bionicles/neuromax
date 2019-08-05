from imageio import imread
import tensorflow as tf
import pandas as pd
import spacy
import json
import os


task_path = os.path.dirname(os.path.realpath('__file__'))
images_path = os.path.join(task_path, "CLEVR_v1.0", "images")
data_path = os.path.join(task_path, "CLEVR_v1.0", "questions",
                         "CLEVR_train_questions.json")

answers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'blue',
           'brown', 'cube', 'cyan', 'cylinder', 'gray', 'green', 'large',
           'metal', 'no', 'purple', 'red', 'rubber', 'small', 'sphere',
           'yellow', 'yes']


spacy.prefer_gpu()
nlp = spacy.load("en_vectors_web_lg")


def get_onehot(answer):
    return tf.convert_to_tensor([0 if x != answers.index(answer) else 1
                                 for x in range(28)])


def get_dataframe():
    global clevr_data
    print("loading clevr JSON file, please wait.")
    clevr_data = json.load(open(data_path))
    print("done loading clevr JSON file")
    print("converting data to dataframe")
    clevr_data = pd.DataFrame(clevr_data["questions"])
    clevr_data.set_index("image_filename", inplace=True)
    question = clevr_data.groupby("image_filename")['question'].apply(lambda x: ','.join(x.astype(str))).reset_index().drop("image_filename", axis=1)
    answer = clevr_data.groupby("image_filename")['answer'].apply(lambda x: ','.join(x.astype(str))).reset_index()
    clevr_data = pd.concat([question, answer], axis=1)


def generate_clevr_item():
    global row_number, question_number
    image_data = clevr_data.loc(row_number)
    image_path = os.path.join(images_path, "train", image_data['image_filename'])
    image_tensor = tf.convert_to_tensor(imread(image_path))
    questions = image_data['question'].split(",")
    question = questions[question_number]
    embedded_question = tf.convert_to_tensor(nlp(question), dtype=tf.float32)
    answer = image_data['answer'].split(",")[question_number]
    one_hot_answer = [get_onehot(answer) for answer in answers]
    row_number += 1
    question_number += 1
    if question_number == len(questions):
        question_number = 0
    return image_tensor, embedded_question, one_hot_answer, question, answer


def get_dataset():
    global row_number, question_number
    row_number, question_number = 0, 0
    get_dataframe()
    return tf.data.Dataset.from_generator(generate_clevr_item,
                                          [tf.int32, tf.float32, tf.int32,
                                           tf.string, tf.string])
