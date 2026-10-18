# MNIST digit classification with SkePU-DNN (translation of SkePU's examples/dnn/dnn_mnist10.cpp).
#
# Usage: dnn_mnist10 DATA_DIR MODEL BATCH_SIZE LEARNING_RATE EPOCHS TRAIN_SAMPLES TEST_SAMPLES BACKEND
#   DATA_DIR: the directory with the MNIST files (examples/data/mnist in SkePU)
#   MODEL: mlp | cnn | conv1d (the image rows as a sequence)
#   TRAIN_SAMPLES, TEST_SAMPLES: use only the first N samples (0 = all)
#
# The neural network is the C++ library of SkePU (skepu::dnn), every step of it a skeleton call; this
# program is the same as the C++ one, built from the layers of skepu.dnn. Translate with --float32.
import skepu
import sys

NUM_CLASSES = 10

def build_mlp(batch):
    model = skepu.dnn.SequentialModel(batch, 28, 28, 1)
    model.add(skepu.dnn.Flatten())
    model.add(skepu.dnn.Dense(128, skepu.dnn.INIT_He_uniform, skepu.dnn.INIT_Zero, skepu.dnn.ACT_ReLU))
    model.add(skepu.dnn.Dense(NUM_CLASSES, skepu.dnn.INIT_Xavier_uniform, skepu.dnn.INIT_Zero, skepu.dnn.ACT_SoftMax))
    return model

def build_cnn(batch):
    model = skepu.dnn.SequentialModel(batch, 28, 28, 1)
    model.add(skepu.dnn.Conv2D(8, 3, 3, skepu.dnn.INIT_He_uniform, skepu.dnn.INIT_Zero, skepu.dnn.ACT_ReLU))
    model.add(skepu.dnn.MaxPooling2D(2, 2))
    model.add(skepu.dnn.Conv2D(16, 3, 3, skepu.dnn.INIT_He_uniform, skepu.dnn.INIT_Zero, skepu.dnn.ACT_ReLU))
    model.add(skepu.dnn.MaxPooling2D(2, 2))
    model.add(skepu.dnn.Flatten())
    model.add(skepu.dnn.Dense(NUM_CLASSES, skepu.dnn.INIT_Xavier_uniform, skepu.dnn.INIT_Zero, skepu.dnn.ACT_SoftMax))
    return model

# Image rows as a sequence: 28 time steps with 28 channels each (input shape batch x 1 x 28 x 28)
def build_conv1d(batch):
    model = skepu.dnn.SequentialModel(batch, 1, 28, 28)
    model.add(skepu.dnn.Conv1D(32, 5, skepu.dnn.INIT_He_uniform, skepu.dnn.INIT_Zero, skepu.dnn.ACT_ReLU))  # 28 -> 24
    model.add(skepu.dnn.MaxPooling1D(2))  # 24 -> 12
    model.add(skepu.dnn.Conv1D(32, 3, skepu.dnn.INIT_He_uniform, skepu.dnn.INIT_Zero, skepu.dnn.ACT_ReLU))  # 12 -> 10
    model.add(skepu.dnn.Flatten())
    model.add(skepu.dnn.Dense(NUM_CLASSES, skepu.dnn.INIT_Xavier_uniform, skepu.dnn.INIT_Zero, skepu.dnn.ACT_SoftMax))
    return model

def main():
    if len(sys.argv) < 9:
        print("Usage:", sys.argv[0], "data_dir model(mlp|cnn|conv1d) batch_size learning_rate epochs train_samples test_samples backend")
        sys.exit(1)

    data_dir = sys.argv[1]
    model_name = sys.argv[2]
    batch_size = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    epochs = int(sys.argv[5])
    max_train = int(sys.argv[6])
    max_test = int(sys.argv[7])
    skepu.set_global_backend_spec(skepu.BackendSpec(sys.argv[8]))

    train_images = skepu.dnn.mnist_images(data_dir + "/train-images.idx3-ubyte", 0.0, 1.0)
    test_images = skepu.dnn.mnist_images(data_dir + "/t10k-images.idx3-ubyte", 0.0, 1.0)
    train_labels = skepu.dnn.mnist_labels(data_dir + "/train-labels.idx1-ubyte")
    test_labels = skepu.dnn.mnist_labels(data_dir + "/t10k-labels.idx1-ubyte")

    n_train = train_images.size_i()
    n_test = test_images.size_i()
    if max_train > 0 and max_train < n_train:
        n_train = max_train
    if max_test > 0 and max_test < n_test:
        n_test = max_test

    if model_name == "conv1d":
        train_x = skepu.dnn.as_sequence(train_images, n_train)
        test_x = skepu.dnn.as_sequence(test_images, n_test)
    else:
        train_x = skepu.dnn.take(train_images, n_train)
        test_x = skepu.dnn.take(test_images, n_test)
    train_y = skepu.dnn.one_hot(train_labels, NUM_CLASSES, n_train)
    test_y = skepu.dnn.one_hot(test_labels, NUM_CLASSES, n_test)

    print("Training on", n_train, "samples, testing on", n_test, "samples")

    if model_name == "mlp":
        model = build_mlp(batch_size)
    elif model_name == "cnn":
        model = build_cnn(batch_size)
    elif model_name == "conv1d":
        model = build_conv1d(batch_size)
    else:
        print("Unknown model '" + model_name + "'")
        sys.exit(1)

    model.init()
    model.configure(skepu.dnn.LOSS_CategoricalCrossEntropy, skepu.dnn.OPT_SGD)
    print(model.summary(), end="")

    model.fit(train_x, train_y, epochs, learning_rate, 0.1)

    confusion = skepu.Matrix(NUM_CLASSES, NUM_CLASSES, 0)
    score = model.evaluate_confusion(test_x, test_y, confusion)
    print("Test loss:", score[0], "Test accuracy:", score[1])
    print("Confusion matrix (rows: actual, columns: predicted):")
    print(confusion)

main()
