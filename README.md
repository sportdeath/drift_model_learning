# Learning Vehicle Drift Dynamics Using Machine Learning

## Training

To train the graph run:

    python3 learn.py

This will periodically save checkpoints in ```tmp/SESSION_NAME/#/```.

## Compiling

Move a particular training checkpoint to the model folder.

    mv tmp/TRAIN_NAME/ITERATION_#/* model/

Construct the execution graph by running

    python3 construct_execution_graph.py

This will save a graph to ```model/f.pbtxt``` and a configuration file to ```model/f.config.pbtxt```

[Install ```bazel```](https://docs.bazel.build/versions/master/install.html) if necessary.
Clone tensorflow and build ```tfcompile``` and ```freeze_graph```.

    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    bazel build tensorflow/compiler/aot:tfcompile
    bazel build tensorflow/python/tools:freeze_graph

Use ```freeze_graph``` to freeze the trained weights into the graph.

    cd drift_model_learning
    ./PATH_TO_TENSORFLOW/bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=model/f.pbtxt --input_checkpoint=mode/model.ckpt --output_graph=model/f_frozen.pbtxt --output_node_names=dstates

Use ```tfcompile``` to compile the frozen graph into C++ code.

    ./PATH_TO_TENSORFLOW/bazel-bin/tensorflow/compiler/aot/tfcompile --graph=model/f_frozen.pbtxt --config=model/f.config.pbtxt --cpp_class="machine_learning_model::f"
