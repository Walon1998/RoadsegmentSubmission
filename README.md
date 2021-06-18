# The Predictors - Submission

This document describes how our code can be used. The report can be found [here!](./report.pdf)

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
* [efficientnet400](./EfficientNet400.py)
* [efficientnet256](./EfficientNet256.py)
* [efficientnet128](./EfficientNet128.py)
* [preefficientnet](./preEfficientNet.py)
* [gc-dcnn](./gc_dcnn.py)

Example:

```bash
python main.py 0.0001 5 4 xception128 simpleTest
```

This will train the Xception 128 model for 5 epochs using batch size 4 with learning rate 0.0001. It will create a new submission called "simpleTest" which can be found in [Predictions](./Predictions)
.

### Usage on euler

To use on euler the following commands can be used.

```bash
module load eth_proxy gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1
bsub -n 8 -W 12:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py <learning rate> <epochs> <batch_size> <model> <name>
```

### Reproducibility

Since we have trained over 150 models and used an ensemble to make our final predictions, it is not possible to easily reproduce our results. However, as we used [Comet](https://www.comet.ml) to keep
track of all our experiments, you can take a look at everything we did, see [here!](https://www.comet.ml/walon1998/roadsegmentation?shareable=BsDLejxcShYvYE3OK6R1gTTMN)

## Create an Ensemble

To create an ensemble, add the model names of which you want to create an ensemble to the `predictions` array in the [create_ensemble.py](./create_ensemble.py) file. Then use the following command

```bash
python create_ensemble.py
```

This will then create a new ensemble in [Predictions](./Predictions) using the combined names of the input models. The predictions we used to create our final submissions are already included.
We have always used the experiment key from [Comet](https://www.comet.ml/walon1998/roadsegmentation?shareable=BsDLejxcShYvYE3OK6R1gTTMN) as the model name.

## Change Dataset

To change the dataset on which the model is trained, go to [main.py](./main.py) and change the training or/and valdition dataset of the `fit` method. Also make sure that the correct dataset is
augmented, see [Datagenerator.py](./Datagenerator.py).

## Create Figures

To recreate the figures used in the report use

```bash
python loss_plot.py
python summary_plot.py
```

## Generate Additional Training Data

To generate additional training data use

```bash
python google_maps_gen.py
```

The python script will propose 100 new possible training images. For each image, 4 possible masks are presented. One can select one of the masks by entering 1,2,3 or in the console and then press
enter. This will add the image to the [Additional Data](./Data/Additional_Data/) directory. If none of the masks are appealing, and you want to skip the image, you can press the Enter key without
entering a number. Make sure that displaying figures is non-blocking in your environment, otherwise it will not be possible to provide input while the plot is displayed (Verified to work in PyCharm).

## GC-DCNN
We have implemented another baseline that unfortunately did not make it into the report. Nevertheless, we want to mention it briefly here.
This approach tries to preserve more global context (GC) data by using kernels with dilation, and a spatial pyramid pooling layer.
Even though in their report, the authors achieved better results than their baselines, with our data it performed about as well as the more established (and simpler) U-Net.
Details can be found in [the paper](https://doi.org/10.1016/j.ins.2020.05.062).










