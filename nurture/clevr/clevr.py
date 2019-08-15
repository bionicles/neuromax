from imageio import imread
import tensorflow as tf
import pandas as pd
import spacy
import json
import os

from tools.get_onehot import get_onehot
from tools.log import log

DATASET = "val"

task_path = os.path.join(".", "nurture", "clevr")
images_path = os.path.join(task_path, "CLEVR_v1.0", "images")
json_data_path = os.path.join(task_path, "CLEVR_v1.0", "questions",
                              f"CLEVR_{DATASET}_questions.json")

csv_data_path = os.path.join(task_path, "CLEVR_v1.0", "questions",
                             f"CLEVR_{DATASET}_questions.csv")

answer_choices = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    'blue', 'brown', 'cube', 'cyan', 'cylinder', 'gray', 'green', 'large',
    'metal', 'no', 'purple', 'red', 'rubber', 'small', 'sphere', 'yellow',
    'yes']


spacy.prefer_gpu()

# install pretrained word vectors:
# python -m spacy download en_vectors_web_lg
# can also be downloaded via https://github.com/explosion/spacy-models/releases
nlp = spacy.load("en_vectors_web_lg")


def run_clevr_task(agent, task_id, task_dict):
    onehot_task_id = get_onehot(task_id, list(agent.tasks.keys()))
    dataset = task_dict.dataset.shuffle(10000)
    model = task_dict.model
    total_free_energy = 0.
    loss = 0.
    for image_tensor, embedded_question, one_hot_answer in dataset.take(task_dict.examples_per_episode):
        inputs = [onehot_task_id, loss, image_tensor, embedded_question]
        inputs = [tf.cast(tf.expand_dims(input, axis=0), dtype=tf.float32)
                  for input in inputs]
        [log(i, input, color="yellow") for i, input in enumerate(inputs)]
        prior_loss_prediction = 0.
        prior_code_prediction = tf.zeros(agent.compute_code_shape(task_dict))
        with tf.GradientTape() as tape:
            normies, code, reconstructions, actions = model(inputs)
            # compute free energy: loss + surprise + complexity - freedom
            one_hot_action = actions[0]
            loss = tf.keras.losses.categorical_crossentropy(
                one_hot_answer, one_hot_action)
            free_energy = agent.compute_free_energy(
                loss=loss, prior_loss_prediction=prior_loss_prediction,
                normies=normies, reconstructions=reconstructions,
                code=code, prior_code_prediction=prior_code_prediction,
                actions=actions
            )
        gradients = tape.gradient([free_energy, model.losses],
                                  model.trainable_variables)
        agent.optimizer.apply_gradients(zip(gradients,
                                            model.trainable_variables))
        total_free_energy += free_energy
    return total_free_energy


def get_dataframe():
    global clevr_data
    if not os.path.exists(csv_data_path):
        print("loading clevr JSON file, please wait.")
        clevr_data = json.load(open(json_data_path))
        print("done loading clevr JSON file")
        print("converting data to dataframe")
        clevr_data = pd.DataFrame(clevr_data["questions"])
        clevr_data.set_index("image_filename", inplace=True)
        question = clevr_data.groupby("image_filename")['question'].apply(
            lambda x: ','.join(x.astype(str))).reset_index().drop("image_filename", axis=1)
        answer = clevr_data.groupby("image_filename")['answer'].apply(
            lambda x: ','.join(x.astype(str))).reset_index()
        clevr_data = pd.concat([question, answer], axis=1)
        clevr_data.to_csv(csv_data_path)
    else:
        print("loading clevr csv file, please wait.")
        clevr_data = pd.read_csv(csv_data_path)
        print("done loading clevr csv file")


def generate_clevr_item():
    global row_number, question_number
    image_data = clevr_data.loc[row_number]
    image_path = os.path.join(images_path, "val", image_data['image_filename'])
    image_tensor = tf.convert_to_tensor(imread(image_path), dtype=tf.float32)
    questions = image_data['question']
    log("image_data['question']", image_data['question'], color="blue")
    questions = questions.split(",")
    log("questions.split(',')", questions, color="blue")
    questions = questions[0]
    log("questions[0]", questions, color="blue")
    questions = questions.split("? ")
    log("questions. split ?", questions, color="blue")
    question = questions[question_number]
    log(question_number, question, color="blue")
    embedded_question = tf.convert_to_tensor(nlp(question).vector,
                                             dtype=tf.float32)
    log(question_number, embedded_question, color="blue")
    answer = image_data['answer'].split(",")[question_number]
    one_hot_answer = get_onehot(answer, answer_choices)
    question_number += 1
    if question_number == len(questions):
        question_number = 0
        row_number += 1
    yield (image_tensor, embedded_question, one_hot_answer)


def read_clevr_dataset():
    global row_number, question_number
    row_number, question_number = 0, 0
    get_dataframe()
    return tf.data.Dataset.from_generator(generate_clevr_item,
                                          (tf.float32, tf.float32, tf.float32))
