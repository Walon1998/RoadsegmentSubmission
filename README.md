# The Predictors - Submission
This document describes how our code can be used. The report can be found [Here!](./report.pdf)

## Install dependencies
All dependencies can be installed using
```bash
pip install -r requirements.txt
```

## Train a model
To train a model use
```bash
python main.py <learning rate> <epochs> <batch_size> <model> <name>
```
where model can be one of the following:
* [xception400](./xception400.py)
* [xception256](./xception256.py)
* [xception128](./xception128.py)
* [prexception](./prexception.py)
* [efficientnet400](./efficientnet400.py)
* [efficientnet256](./efficientnet256.py)
* [efficientnet128](./efficientnet128.py)
* [preefficientnet](./preefficientnet.py)

Example:
```bash
python main.py 0.0001 5 4 xception128 simpleTest
```
This will train the Xception 128 model for 5 epochs using batch size 4 with learning rate 0.0001. It will create a new submission called "simpleTest" which can be found in [Predictions](./Predictions).

### Usage on euler
To use on euler the following commands can be used.
```bash
module load eth_proxy gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1
bsub -n 8 -W 12:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py <learning rate> <epochs> <batch_size> <model> <name>
```

### Reproducibility
Since we have trained over 150 models and used an ensemble to make our final predictions, it is not possible to easily reproduce our results.
However, as we used [Comet](https://www.comet.ml) to keep track of all our experiments, you can take a look at everything we did, see [here!](https://www.comet.ml/walon1998/roadsegmentation?shareable=BsDLejxcShYvYE3OK6R1gTTMN)


## Create an Ensemble
To create an ensemble, add the model names of which you want to create an ensemble to the `predictions` array in the [create_ensemble.py](./create_ensemble.py) file.
Then use the following command
```bash
python create_ensemble.py
```
This will then create a new ensemble in [Predictions](./Predictions) using the combined names of the input models.
The predictions we used to create our final submissions are already included in [predictions](./predictions).
We have always used the experiment key from [Comet](https://www.comet.ml/walon1998/roadsegmentation?shareable=BsDLejxcShYvYE3OK6R1gTTMN) as the model name.










