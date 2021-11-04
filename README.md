# DGCIT: Double Generative Adversarial Networks for Conditional Independence Testing

This repository contains an implementation and further details of Double Generative Adversarial Networks for Conditional Independence Testing.

Reference: Shi, C., Xu, T., Bergsma, W. and Li, L. (2021+) Double Generative Adversarial Networks for Conditional Independence Testing. Journal of Machine Learning Research, accepted.

Paper link: https://arxiv.org/pdf/2006.02615.pdf 

## Setup

```
$ git clone https://github.com/tianlinxu312/dgcit.git
$ cd dgcit/

# Create a virtual environment called 'venv'
$ virtualenv venv 
$ source venv/bin/activate    # Activate virtual environment

# Install all dependencies
$ python3 -m pip install -r requirements.txt 
```

## Data
CCLE data used in the paper is downloaded from here: https://github.com/alexisbellot/GCIT/tree/master/CCLE%20Experiments

## Training 
For runing experiments to compute Type I error: 

```
# Compute Type I error for 1000 samples
$ python3 train.py \
    --model="dgcit"
    --test="type1error"
    --n_samples=1000
```

For runing experiments to compute Power:  
```
# Compute Power for 1000 samples
$ python3 train.py \
    --model="dgcit"
    --test="power"
    --n_samples=1000
```
For more baseline models and parameter settings, please see `train.py` file.  


## Illustration of conditional independence testing with double GANs
<img src="./figs/dgcit.png" width="750" alt="dgct">

## Type I error and power
<img src="./figs/tp.png" width="750" alt="tp">
Top panels: the empirical type-I error rate of various tests under H0. From left to right:
normal Z with α = 0.1, normal Z with α = 0.05, Laplacian Z with α = 0.1, and Laplacian Z
with α = 0.05. Bottom panels: the empirical power of various tests under H1.

KCIT results were obtained using this implementation: https://github.com/ericstrobl/RCIT/blob/master/R/KCIT.R

## Results for the anti-cancer drug example
<img src="./figs/table.png" width="750" alt="tp">
The variable importance measures of the elastic net(EN) and random forest(RF) models, versus the
p-values of the GCIT and DGCIT tests for the anti-cancer drug example.
 

