import tensorflow as tf

def divide(t):
    return t / 5

@tf.function
def autograph_isnt_auto(t):
  while t < 100 * 3.14 and t  > -1:
    t = t + 1
    t = divide(t)
  return t

tensor = 1
t = autograph_isnt_auto(tensor)
tf.print(t)
