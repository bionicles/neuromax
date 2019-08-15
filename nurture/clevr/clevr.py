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
        expanded_loss = tf.expand_dims(loss, 0)
        inputs = [onehot_task_id, expanded_loss, image_tensor, embedded_question]
        inputs = [tf.cast(tf.expand_dims(input, axis=0), dtype=tf.float32)
                  for input in inputs]
        prior_code_prediction = tf.zeros(agent.compute_code_shape(task_dict))
        prior_loss_prediction = 0.
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            [log(output, color="blue") for output in outputs]
            normies, code, reconstructions, actions = agent.unpack(
                outputs, task_dict)

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
        clevr_data.to_csv(csv_data_path)
    else:
        print("loading clevr csv file, please wait.")
        clevr_data = pd.read_csv(csv_data_path)
        print("done loading clevr csv file")


def generate_clevr_item():
    global row_number
    image_data = clevr_data.loc[row_number]
    image_path = os.path.join(images_path, "val", image_data['image_filename'])
    image_tensor = tf.convert_to_tensor(imread(image_path), dtype=tf.int32)
    image_tensor = tf.einsum("WHC->HWC", image_tensor)
    question = image_data['question']
    question = nlp(question)
    embedded_question = tf.convert_to_tensor(question.vector)
    tensors = [tf.convert_to_tensor(word.vector) for word in question]
    tensors.append(embedded_question)
    tensors = [tf.expand_dims(tensor, 0) for tensor in tensors]
    question_embedding = tf.concat(tensors, axis=0)
    answer = image_data['answer']
    one_hot_answer = get_onehot(answer, answer_choices)
    row_number += 1
    yield (image_tensor, question_embedding, one_hot_answer)


def read_clevr_dataset():
    global row_number
    row_number = 0
    get_dataframe()
    return tf.data.Dataset.from_generator(generate_clevr_item,
                                          (tf.float32, tf.float32, tf.float32))
