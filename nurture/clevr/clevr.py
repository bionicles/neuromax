import tensorflow as tf
import random
# import spacy  # don't forget to python -m spacy download en
# import wmd
#
# nlp = spacy.load("en_core_web_md")
# nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)

answers = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    'blue',
    'brown',
    'cube',
    'cyan',
    'cylinder',
    'gray',
    'green',
    'large',
    'metal',
    'no',
    'purple',
    'red',
    'rubber',
    'small',
    'sphere',
    'yellow',
    'yes']
i = answers.index(answer)
y_true = tf.convert_to_tensor([0 if x != i else 1 for x in range(28)])

# def wmd_loss(answers, y_true, y_pred):
#     index = tf.where(y_pred)[0][0]
#     print(index)
#     y_pred = answers[index]
#     print(y_pred)
#     y_true = nlp(y_true)
#     y_pred = nlp(y_pred)
#     print(y_true)
#     print(y_pred)
#     similarity = y_true.similarity(y_pred)
#     loss = -1 * similarity
#     return tf.convert_to_tensor(loss, tf.float32)
#
#
# print(len(answers), "answers")
#
answer = random.choice(answers)



y_pred = tf.random.uniform((28,), dtype=tf.float32)
y_pred /= tf.math.reduce_sum(y_pred)

print(answer, i, y_true)
print(y_pred, tf.math.reduce_sum(y_pred))

loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
print(loss)



# tf.print(y_pred)
#
# y_true = answers[random.randint(0, 28)]
# answers2 = tf.convert_to_tensor(answers, tf.string)
# wmd_loss(answers2, y_true, y_pred)
