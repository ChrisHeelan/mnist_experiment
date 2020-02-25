# MNIST Experiment

This repository contains a [Dockex](https://github.com/ConnexonSystems/dockex) experiment for performing 
classification on the MNIST dataset.

This repository is under active development. Check back for new modules and documentation.

## Demo:

![Dockex MNIST Experiment progress](docs/dockex_mnist_experiment_progress.gif)

The GIF above shows a Dockex cluster running on Google Cloud Platform composed of 16 VMs with 192 CPU cores, 
759 GB of RAM, and 16 GPUs (4 T4's, 4 P100's, and 8 K80's). An MNIST experiment is executed that trains and evaluates 
 99 logistic regression models, 135 k-nearest neighbor models, 33 random forest models, and 54 convolutional neural 
 network models. The GIF is rendered at 2x actual speed.

## How to use:

1. Install and run [Dockex](https://github.com/ConnexonSystems/dockex).

2. Clone this repository.

3. If Dockex was installed with GPU support, set the ENABLE_GPU flag to ```True``` in ```experiments/mnist_experiment.py```.

4. Use the Dockex GUI MACHINES tab to set CPU credits > 0. If using GPU support, also set GPU credits to 1.

5. Use the Dockex GUI LAUNCH tab to launch the experiment with the following:

* Project Path: ```/path/to/mnist_experiment```

* Experiment Path: ```experiments/mnist_experiment.py```

Results will be written to the tmp_dockex_path (defaults to ```/tmp/dockex/data```).
