# Monitoring PyTorch Training with Tensorboard

To start training use:

    cd examples/
    python3 train.py
 
The script will start training a simple CNN on CIFAR10 and use
Tensorboard to create log files. These log files can be viewed using:
 
a) Tensorboard itself:

    python3 -m tensorboard.main --logdir=./checkpoints/logs/ --host=localhost --port=6006
     
Unfortunately, Tensorboad does not allow to easily export the created plots.
Also, it might be required to directly access the logs and create
plots using other libraries.

![Tensorboard](screenshots/tensorboard.png?raw=true "Tensorboard")

b) Using a custom Jupyter Notebook based on [lab-ml/labml](https://github.com/lab-ml/labml)
which can be found in `examples/monitor.ipynb`.

![Lab-ML](screenshots/labml.png?raw=true "Lab-ML")