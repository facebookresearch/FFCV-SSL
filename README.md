<p align = 'center'>
<em><b>Fast Forward Computer Vision</b>: train models at a fraction of the cost with accelerated data loading!</em>
</p>
<img src='assets/logo.svg' width='100%'/>
<p align = 'center'>
<!-- <br /> -->
[<a href="#install-with-anaconda">install</a>]
[<a href="#quickstart">quickstart</a>]
[<a href="#features">features</a>]
[<a href="https://docs.ffcv.io">docs</a>]
[<a href="https://join.slack.com/t/ffcv-workspace/shared_invite/zt-11olgvyfl-dfFerPxlm6WtmlgdMuw_2A">support slack</a>]
[<a href="https://ffcv.io">homepage</a>]
<br>
Maintainers:
<a href="https://twitter.com/gpoleclerc">Guillaume Leclerc</a>,
<a href="https://twitter.com/andrew_ilyas">Andrew Ilyas</a> and
<a href="https://twitter.com/logan_engstrom">Logan Engstrom</a>
</p>

`ffcv` is a drop-in data loading system that dramatically increases data throughput in model training:

- [Train an ImageNet model](#prepackaged-computer-vision-benchmarks)
on one GPU in 35 minutes (98¢/model on AWS)
- [Train a CIFAR-10 model](https://docs.ffcv.io/ffcv_examples/cifar10.html)
on one GPU in 36 seconds (2¢/model on AWS)
- Train a `$YOUR_DATASET` model `$REALLY_FAST` (for `$WAY_LESS`)

Keep your training algorithm the same, just replace the data loader! Look at these speedups:

<img src="assets/headline.svg" width='830px'/>

`ffcv` also comes prepacked with [fast, simple code](https://github.com/libffcv/imagenet-example) for [standard vision benchmarks]((https://docs.ffcv.io/benchmarks.html)):

<img src="docs/_static/perf_scatterplot.svg" width='830px'/>

# Installation
```
conda create -y -n ffcv python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
conda activate ffcv
pip install ffcv
```
Troubleshooting note: if the above commands result in a package conflict error, try running ``conda config --env --set channel_priority flexible`` in the environment and rerunning the installation command.

# What's new <a name="introduction"></a>

- *More augmentations*: Data augmentations is crucial for methods like Self-supervised learning, in this fork we add ColorJitter, Solarization, Grayscale, Rotation.. 
- *Seeding** Being able to fix the seed of a given transformation is important for reproducibility
- *Multi views* FFCV2 is able to return an arbitraty number of different view of a given field with different pipelines.

# `ffcv` seeding <a name="seeding"></a>
 
To reproduce those figures, please run [this code](./examples/test_ffcv_augmentations_seeding.py) located in the examples folder.
By default the loader and DAs have a `None` seed i.e. different runs have different data ordering and augmentaiton realizations as shown below for a mini-batch size of 3 on Imagenet only with random crop and translation as DAs. 
For reproducibility we enable seed specification for both the loader and each DA leading to the follow where both are set


DAs and loader with seed=None             |  DAs and loader with seed=0
:-------------------------:|:-------------------------:
![](./assets/visual_images_loaderNone_DANone.png) |  ![](./assets/visual_images_loader0_DA0.png)


which allows for fine-control e.g. setting only the seed of the DA to `0` leaves the data ordering independent between runs

DAs with seed=0             |  DAs with seed=1
:-------------------------:|:-------------------------:
![](./assets/visual_images_loaderNone_DA0.png) |  ![](./assets/visual_images_loaderNone_DA1.png)



and conversely setting only the seed of the loader leaves the DA independent between runs
which again can be modified by hand to manually change the data ordering as seen comparing the below figures
loader with seed=0             |  loader with seed=1
:-------------------------:|:-------------------------:
![](./assets/visual_images_loader0_DANone.png) |  ![](./assets/visual_images_loader1_DANone.png)



# `ffcv.transforms.image`<a name="image_augs"></a>

All the plots here use the same seed for the data-augmentations hence the same realisation. To reproduce those figures use [this code](../examples/test_ffcv_augmentations_families.py).

*RandomCropResized* with loader seed=None             |  with loader seed=0
:-------------------------:|:-------------------------:
![](./assets/visual_images_resizedcrop_None.png) |  ![](./assets/visual_images_resizedcrop_0.png)


*RandomColorJitter* with loader seed=None             |  with loader seed=0
:-------------------------:|:-------------------------:
![](./assets/visual_images_jitter_None.png) |  ![](./assets/visual_images_jitter_0.png)


*RandomErasing* with loader seed=None             |  with loader seed=0
:-------------------------:|:-------------------------:
![](./assets/visual_images_erasing_None.png) |  ![](./assets/visual_images_erasing_0.png)


*RandomTranslation* with loader seed=None             |  with loader seed=0
:-------------------------:|:-------------------------:
![](./assets/visual_images_translate_None.png) |  ![](./assets/visual_images_translate_0.png)


*RandomSolarization* with loader seed=None             |  with loader seed=0
:-------------------------:|:-------------------------:
![](./assets/visual_images_solarization_None.png) |  ![](./assets/visual_images_solarization_0.png)


*RandomGrayscale* with loader seed=None             |  with loader seed=0
:-------------------------:|:-------------------------:
![](./assets/visual_images_grayscale_None.png) |  ![](./assets/visual_images_grayscale_0.png)

*RandomHorizontalFlip* with loader seed=None             |  with loader seed=0
:-------------------------:|:-------------------------:
![](./assets/visual_images_horizontal_None.png) |  ![](./assets/visual_images_horizontal_0.png)

*RandomVerticalFlip* with loader seed=None             |  with loader seed=0
:-------------------------:|:-------------------------:
![](./assets/visual_images_vertical_None.png) |  ![](./assets/visual_images_vertical_0.png)

*RandomInvert* with loader seed=None             |  with loader seed=0
:-------------------------:|:-------------------------:
![](./assets/visual_images_invert_None.png) |  ![](./assets/visual_images_invert_0.png)


# Multiview augmentation e.g. for SSL<a name="multiview"></a>

To reproduce those figures use [this code](../examples/test_ffcv_augmentations_ssl.py). We show how to produce three views from a dataset generated with only two fields: `image` and `label`. In short, we use the loader's `custom_field_mapper` argument to give a dictionnary mapping the extra given pipelines to the ones present in the saved dataset. In this case we also control the seed of the grayscale augmentation so that it is the same realisations between `view1` and `view2`, leading to the below plot:

![](./assets/visual_images_ssl.png)



# How to contribute <a name="contribute"></a>



# References for the implementations


- Graph Laplacian Estimation:
  - [TODO](https://github.com/rodrigo-pena/graph-learning)
  -  [SGL](https://github.com/anshul3899/Structured-Graph-Learning)
- [COke](https://github.com/idstcv/CoKe) for augmentations and losses:
  - `torchstrap.torchvision_pipelines.coke_imagenet_single_view`
  - `torchstrap.torchvision_pipelines.coke_imagenet_double_view`
  - `torchstrap.torchvision_pipelines.coke_imagenet_multi_view`
  - `torchstrap.criterion.coke_single_view`
  - `torchstrap.criterion.coke_multi_view`
-


## License

This project is released under MIT License, which allows commercial use. See [LICENSE](LICENSE) for details.
