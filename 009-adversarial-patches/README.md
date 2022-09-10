# Adversarial Patches and Frames

Run

    cd examples/
    python3 attack.py --attack=patches --size=8 --model_file=path_to_model

to compute adversarial patches of size `8` for the provided model. Patches are
constrained not to occur in the center of the image. `--attack=frames` can be
used to generate adversarial frames in which case `size` is the border size
in pixels.