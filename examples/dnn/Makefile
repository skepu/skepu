# The SkePU-DNN examples (see the top of ../Makefile). The programs read their data
# sets from ../data and the models from models/.
PROGS = dnn_binary dnn_artificial dnn_cifar10 dnn_mnist10 dnn_model_file dnn_signals1d
COMMON_SOURCES =

# Arguments of 'make run/<program>'
lookup.dnn_binary     = 32 0.05 1 $(RUN_BACKEND) 640 128
lookup.dnn_artificial = 32 0.05 2 $(RUN_BACKEND)
lookup.dnn_cifar10    = small 32 0.001 1 $(RUN_BACKEND) 1600 330 --adam
lookup.dnn_mnist10    = cnn 32 0.05 1 $(RUN_BACKEND) 1600 330
lookup.dnn_model_file = models/mnist_cnn.dnn 32 0.05 2 $(RUN_BACKEND) 3200 640 trained.dnn
lookup.dnn_signals1d  = conv1d 32 0.003 2 $(RUN_BACKEND) 1600 330 --adam

include ../Makefile
