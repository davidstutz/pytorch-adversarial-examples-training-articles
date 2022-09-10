# Distal Adversarial Examples

Using one of the models of `5-adversarial-training` or `6-confidence-calibrated-adversarial-training`, use

    cd examples/ 
    python3 attack.py --model=model_path

to evaluate how well distal adversarial examples can be detected based on their confidence. Poor false positive rate
indicates that the network gives high confidence on rubbish inputs.
