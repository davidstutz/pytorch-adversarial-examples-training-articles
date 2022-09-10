# L_p Adversarial Examples on Cifar10

Using one of the models from `003-autoaugment-cifar10`,
available for download [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/PrsSqnXHD2RyMfG), run

    cd examples/
    python3 attack.py --attack=linf --model=model_path
    
to attack the model using `linf`, `l2`, or `l1` projected gradient descent
(PGD) attacks.