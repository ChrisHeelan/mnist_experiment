# MNIST Experiment

This repository contains a [Dockex](https://github.com/ConnexonSystems/dockex) experiment for performing 
classification on the MNIST dataset.

This repository is under active development. Check back for new modules and documentation.

## How to use:

1. Install [Dockex](https://github.com/ConnexonSystems/dockex).

2. If not using GPUs, set the ENABLE_GPU flag to ```False``` in ```experiments/mnist_experiment.py```.

3. Use the Dockex GUI MACHINES tab to set CPU credits > 0. If using GPU support, also set GPU credits to 1.

4. Use the Dockex GUI LAUNCH tab to launch the experiment with the following:

* Project Path: ```/path/to/mnist_experiment```

* Experiment Path: ```experiments/mnist_experiment.py```

Results will be written to the tmp_dockex_path (defaults to ```/tmp/dockex/data```).
