# Learning Vehicle Drift Dynamics Using Machine Learning

## Training

To train the graph run:

    python3 learn.py

This will periodically save checkpoints in ```tmp/LOG_DIR/SESSION_NAME/ITERATION_#/```.
Visualize the training by running

    tensorboard --logdir=tmp/LOG_DIR

## Compiling

Move a particular training checkpoint to the model folder.

    mv tmp/LOG_DIR/SESSION_NAME/ITERATION_#/* model/

Construct the execution graph by running

    python3 construct_execution_graph.py

This will save a graph to ```model/f.pbtxt``` and a configuration file to ```model/f.config.pbtxt```

[Install ```bazel```](https://docs.bazel.build/versions/master/install.html) if necessary.
Clone [```tensorflow```](https://github.com/tensorflow/tensorflow) and build [```tfcompile```](https://www.tensorflow.org/performance/xla/tfcompile) and [```freeze_graph```](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py).

    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    bazel build tensorflow/compiler/aot:tfcompile
    bazel build tensorflow/python/tools:freeze_graph

Use ```freeze_graph``` to freeze the trained weights into the graph.

    cd drift_model_learning
    ./PATH_TO_TENSORFLOW/bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=model/f.pbtxt --input_checkpoint=model/model.ckpt --output_node_names=dstates --output_graph=model/f_frozen.pb


Use ```tfcompile``` to compile the frozen graph into C++ code.

    ./PATH_TO_TENSORFLOW/bazel-bin/tensorflow/compiler/aot/tfcompile --graph=model/f_frozen.pb --config=model/f.config.pbtxt --cpp_class="machine_learning_model::f" --out_header="cpp/out.h" --out_metadata_object="cpp/out_helper.o" --out_function_object="cpp/out_model.o"
