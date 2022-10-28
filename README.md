# Table of contents
1. [Why torchstrap](#introduction)
2. [code templating](#templating)
3. [trainer pipeline](#pipeline)
4. FFCV augmentations, seeding and multiview
   - [data augmentation and loader seeding](#seeding)
   - [image augmentations](#image_augs)
   - [multi-view for SSL](#multiview)
5. [Installation](#installation)
6. [How to contribute](#contribute)


# Why torchstrap <a name="introduction"></a>


The goal of torchstrap is to provide an API-like utility for AI/deep learning researchers to enable ultra-fast testing of any ideas that roam around.
Torchstrap aims at ``smartly combining'' minimal features from a few successful libraries while compartimenting them to enable anyone to use only the desired torchstrap features without the others interefering:

- *pytorch-lightning*: we use a similar (but lightweight) `torchstrap.Trainer` wih user-defined function calls before each train/val epoch/step e.g. `Trainer.before_eval_step` but without all the ``behind the curtains magic'' and with a stronger focus on research rather than production so it is simple to implement its own `Trainer.create_model` or `Trainer.initialize_optimizer` and so on
- *fastargs+argparse*: we use a rich but lightweight templating strategy so that all hyper-parameters/function arguments are tracked and accessible/exportable in a few lines for examples simply define a function with
    ```
    @torchstrap.config.param("optimizer.lr")
    def my_fn(lr):
        ...
    ```
    and the value of `lr` can be either given by hand e.g. calling `my_fn(1.0)` or can be loaded from a config file, or given from command line `python main.py --optimizer.lr 1.0` and its value will be logged as part of the hyper-parameters, more details in [this section](#templating)
- *pytables/pyml/json/hdf5*: when doing research, logging is crucial, and it is not only about simple scalars... which is why we build-upon highly efficient and scalable h5 logging for arrays, and simpler txt/json logging for everything else; each version can be called with the values to log e.g. `Trainer.log({"loss":loss.item(),"lr":self.optimizer.lr.item()})` and `Trainer.log_hdf5({"layer1w":self.model.layer1.weight.detach().numpy()})` which allows to keep track of desired value throughout traiing
- *pytorch-metric-learning+torch.nn*: we provide out-of-the-box the criterions offered from those two libraries which can be used and parametrized directly from the templating scheme
- *FFCV*: speed is crucial to quickly try ideas which is why we provide a rich augmented version of FFCV with seeding, more augmentations, multi-view support and so on and so forth, more details in [this section](#seeding)
- *submitit*: when working on a cluster it is crucial to submit jobs, requeue, save temporary checkpoints... which is why we interfaced `Trainer` with a submitit wrapper so that it all gets handled automatically

# Fast code templating <a name="templating"></a>

The key idea is that we do not want to carry around our hyper-parameters... hence leveraging the powerful `fastargs` library, you can now position anywhere in your code the following (example)

```
from fastargs import Param, Section
from fastargs.validation import OneOf


Section("custom_name", "brief description if needed").params(
    my_param1=Param(
        OneOf([0.0,0.5,1.0]),
        "specific description if needed",
        required=True
    ),
    my_param2=Param(
        OneOf(["train", "test"]),
        "another specific description if needed",
        default="ConstantLR",
    )
)
```
and then wherever you need to access those hyper-parameter within your codebase, simply do (for example)
```
from fastargs.decorators import param
@param("model.load_from")
def load_checkpoint(self, load_from: str):
    """load a model and optionnally a scheduler/optimizer/epoch from a given path to checkpoint

    Args:
        load_from (str): path to checkpoint
    """
```


# Workflow of `torchstrap.Trainer` <a name="pipeline"></a>

```bash

# ----- INIT (__init__ method) ------

initialize_model() # and then do a bunch of stuff to change device, dtype, distributed, sync_BN, ....
if not training.eval_only:
    initializer_train_loader()
initializer_val_loader()
if not training.eval_only:
    initialize_optimizer()
    initialize_scheduler()

load_checkpoint()
initialize_metrics()
initialize_criterion()
initialize_logger()

# ---- TRAINING LOOP (train_all_epochs method) ------

for epoch in epochs
    
    modules.train() # put all children modules in train mode
    before_train_epoch() # <- nothing by default
    for step, batch in enumerate(train_dataset)
        self.data=batch and self.step=step # <- for easy access
        before_train_step() # <- nothing by default
        train_step() # <- forward + loss + backward + gradient step + scheduler if not MultiStepLR or StepLR
        after_train_step() # <- nothing by default
    update scheduler if  MultiStepLR or StepLR
    after_train_epoch() # <- nothing by default

    # ---- (only if logging.eval_each_epoch == True) ----
    
    modules.eval() # put all children modules in eval mode
    before_eval_epoch() # <- nothing by default
    for batch in eval_dataset
        self.data=batch and self.step=step # <- for easy access
        before_eval_step() # <- nothing by default
        eval_step() # w/ test time augs. and record metrics
        after_eval_step() # <- nothing by default
    after_eval_epoch() # <- nothing by default
```
# `torchstrap.data.ffcv` seeding <a name="seeding"></a>
 
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



# `torchstrap.data.ffcv.transforms.image`<a name="image_augs"></a>

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



# Installation <a name="installation"></a>

simply do `pip install .` in the repository

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
