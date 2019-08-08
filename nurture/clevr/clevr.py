from imageio import imread
import tensorflow as tf
import pandas as pd
import spacy
import json
import os

from tools import get_onehot

task_path = os.path.dirname(os.path.realpath('__file__'))
images_path = os.path.join(task_path, "CLEVR_v1.0", "images")
data_path = os.path.join(task_path, "CLEVR_v1.0", "questions",
                         "CLEVR_train_questions.json")

answer_choices = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    'blue', 'brown', 'cube', 'cyan', 'cylinder', 'gray', 'green', 'large',
    'metal', 'no', 'purple', 'red', 'rubber', 'small', 'sphere', 'yellow',
    'yes']


spacy.prefer_gpu()

# install pretrained word vectors:
# python -m spacy download en_vectors_web_lg
nlp = spacy.load("en_vectors_web_lg")


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
    one_hot_answer = get_onehot(answer, answer_choices)
    row_number += 1
    question_number += 1
    if question_number == len(questions):
        question_number = 0
    return image_tensor, embedded_question, one_hot_answer


def read_clevr_dataset():
    global row_number, question_number
    row_number, question_number = 0, 0
    get_dataframe()
    return tf.data.Dataset.from_generator(generate_clevr_item,
                                          [tf.int32, tf.float32, tf.int32,
                                           tf.string, tf.string])


def run_clevr_task(agent, task_key, task_dict):
    dataset = task_dict.dataset.shuffle(10000)
    model = agent.models[task_key]
    total_free_energy = 0.
    for image_tensor, embedded_question, one_hot_answer in dataset.take(task_dict.examples_per_episode):
        inputs = [image_tensor, embedded_question]
        with tf.GradientTape() as tape:
            normies, codes, reconstructions, state_predictions, loss_prediction, actions = \
                model(inputs)
            # compute free energy: loss + surprise + complexity - freedom
            one_hot_action = actions[0]
            loss = tf.keras.losses.categorical_crossentropy(
                one_hot_answer, one_hot_action)
            reconstruction_surprise = tf.math.sum([
                -1 * belief.log_prob(truth)
                for belief, truth in zip(reconstructions, normies)])
            loss_surprise = -1 * loss_prediction.log_prob(loss)
            surprise = reconstruction_surprise + loss_surprise
            # how do you measure complexity?
            # maybe L1/L2 reg or entropy/KL
            freedom = actions[0].entropy()
            free_energy = loss + surprise - freedom
        gradients = tape.gradient([free_energy, model.losses],
                                  model.trainable_variables)
        agent.optimizer.apply_gradients(zip(gradients,
                                            model.trainable_variables))
        total_free_energy += free_energy
    return total_free_energy
