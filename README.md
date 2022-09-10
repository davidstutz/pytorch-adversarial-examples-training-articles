## PyTorch Examples

This repository contains several PyTorch examples demonstrating adversarial training, confidence-calibrated adversarial training as well as adversarial robustness evaluation in PyTorch:

* [Monitoring PyTorch Training with Tensorboard](001-monitoring-training/README.md)
* [Easily Saving and Loading PyTorch Models](002-loading-models/README.md)
* [2.56% on Cifar10 with AutoAugment](003-autoaugment-cifar10/README.md)
* [L_p Adversarial Examples on Cifar10](004-pgd/README.md)
* [Adversarial Training on Cifar10](005-adversarial-training)
* [Confidence-Calibrated Adversarial Training on Cifar10](006-confidence-calibrated-adversarial-training/README.md)
* [Proper Robustness Evaluation](007-robustness-evaluation/README.md)
* [Distal Adversarial Examples](008-distal-adversarial-examples/README.md)
* [Adversarial Patches and Frames](009-adversarial-patches/README.md)
* [Adversarial transformations](010-adversarial-transformations/README.md)

The examples correspond to an article series on my blog:
[davidstutz.de](https://davidstutz.de/).

Training, monitoring and attack modules are highly flexible and can easily be
adapted and extended for custom research projects.

Note that large parts of this repository are taken from my latest
ICML'20 [1] and ICCV'21 papers [2], including the repository
[davidstutz/confidence-calibrated-adversarial-training](https://github.com/davidstutz/confidence-calibrated-adversarial-training/),
as well as my student's ECCV'21 workshop paper [3].
    
    [1] D. Stutz, M. Hein, B. Schiele.
        Confidence-Calibrated Adversarial Training: Generalizing to Unseen Attacks.
        ICML, 2020.
    [2] D. Stutz, M. Hein, B. Schiele.
        Relating Adversarially Robust Generalization to Flat Minima.
        ICCV, 2021.
    [3] S. Rao, D. Stutz, B. Schiele.
        Adversarial Training Against Location-Optimized Adversarial Patches
        ECCV Workshops, 2020.

## Installation

Installation is easy with [Conda](https://docs.conda.io/en/latest/):

    conda env create -f environment.yml

You can use `python3 setup.py` to check some of the requirements.
See `environment.yml` for details and versions.

## License

 Copyright (c) 2022 David Stutz, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.