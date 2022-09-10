# Easily Saving and Loading PyTorch Models

Run

    cd examples/
    python3 save_load.py
    
to test the `common.state` module. This module allows to load PyTorch models
without pre-defining the model (and knowing the architecture).

This is accomplished through the structure in `models/classifier.py`: 
`common.state.State` saves not only the model's state dict but also 
the model's class and it's attributes (except private ones prefixed with `__`).
When loading a state file, `common.state.State` first extracts the class name
and the attributes. The model's attributes are, at the same time, arguments
to the constructor.