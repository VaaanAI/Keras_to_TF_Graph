import os
import cv2
import time
import numpy as np
import tensorflow as tf

"""
Author: Shivam Dixit
Email: shivam.dixit@vaaaninfra.com
"""

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

frozen_model = './PATH/TO/FROZEN_MODEL(.pb file)' ## .pb file

# Load frozen graph using TensorFlow 1.x functions
with tf.io.gfile.GFile(frozen_model, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())


# Wrap frozen graph to ConcreteFunctions
frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["x:0"],
                                outputs=["Identity:0"],
                                print_graph=False)

# print("-" * 50)
# print("Frozen model inputs: ")
# print(frozen_func.inputs)
# print("Frozen model outputs: ")
# print(frozen_func.outputs)

path = 'source_dir' ## Path to the dir with all the Images

labels_name = {0:'Class_1', 1:'Class_2', ............. n:'Class_n'}

IMAGES = [os.path.join(path, img) for img in os.listdir(path) if img.endswith('.jpg')]
total_time = 0
img_count = 0
for image in IMAGES:

    im = cv2.imread(image)

    img_count += 1

    start = time.time()

    ## Image size should be according to the input layer of the model
    im = cv2.resize(im, (224, 224)).astype('float32') ## (224, 224) for ResNet50V2

    img = np.expand_dims(im, axis=0)

    # Get predictions for test images
    frozen_graph_predictions = frozen_func(x=tf.constant(img))[0]

    end = time.time()
    time_ = end-start

    total_time += time_

    print(f"{labels_name[tf.argmax(frozen_graph_predictions[0].numpy()).numpy()]}, Classification time: {time_} Sec")

print('\nTotal image count:', img_count)
print(f'\nAvg time per image: {round(total_time/img_count, 4)}')

