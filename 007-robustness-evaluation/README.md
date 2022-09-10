# Proper Robustness Evaluation

Compared to `004-pgd`, running PGD only once with few iterations is generally
_not_ sufficient to evaluate robust models, such as those trained in
`5-adversarial-training` or `6-confidence-calibrated-adversarial-training`.

Using one of the models of `5-adversarial-training` or `6-confidence-calibrated-adversarial-training`, use

    cd examples/ 
    python3 attack.py --model=model_path --attacks=attack1,attack2

to evaluate robustness using multiple random restarts, more iterations and
varying hyper-parameters. Supported attacks are `pgd_linf|l2|l1`,
`pgd_confg_linf|l2|l1` which maximizes confidence, `aa_linf|l2` corresponding to
AutoAttack, and `aa_conf_linf|l2` for AutoAttack maximizing confidence.