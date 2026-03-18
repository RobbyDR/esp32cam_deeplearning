import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

ops = interpreter._get_ops_details()
used_ops = set([op['op_name'] for op in ops])
print(used_ops)