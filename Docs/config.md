## configuration file usage

In general, you only need to customize module parameters within this configuration file to complete ILT tasks. Alternatively, you can choose not to configure anything and use the default parameters. The current default configurations can be found in [config-file](../mtilt/config/defaults.py). Furthermore, you can also extend it to add more features to meet your requirements. You can refer to [doing things in new ways](https://detectron2.readthedocs.io/en/latest/tutorials/extend.html). 

Below, we provide configuration explanations for key modules:

* Determine and configure [lithographic model](../mtilt/solving/litho_operator/build.py)

```yaml
LITHO_OPERATOR:
  NAME: "BasicLithoSim"
  KERNEL_NUM: 24
  PRINT_THRESH: 0.5
  PRINT_STEEPNESS: 50.0
  DOSE_NOM: 1.0
  DOSE_MIN: 0.98
  DOSE_MAX: 1.02
```
    
Here, you need to verify your lithography model parameters, such as `target_density` or `kernel_num`. For more parameters, please refer to [LITHO_OPERATOR](../mtilt/config/defaults.py#L72)

* Determine and configure [initializer](../mtilt/solving/intializer/build.py)
```yaml
INITIALIZER:
  NAME: "PixelInitializer" # or 'LevelSetInitializer'
  TYPE: "pixel" # or 'levelset'
```
Here, you need to confirm your initialization method. This library provides two options: [Pixel-initializer](../mtilt/solving/intializer/pixel_initializer.py) & [levelSet-initializer](../mtilt/solving/intializer/leavelset_initializer.py). 

* Determine and configure [algorithm](../mtilt/solving/algorithms/build.py)
```yaml
ALGORITHM:
  NAME: "GeneralAlgorithm"
  OPTIMIZER: "SGD"
```
Here, You need to specify the optimization algorithm and optimizer. This repository defaults to using the `SGD` optimizer for iteratively updating the mask. Alternatively, you can choose multi-objective optimization. [Weighting Methods](mtilt/optimizer/weighting). You can directly specify MOO in configuration file:
```yaml
ALGORITHM:
  NAME: "GeneralAlgorithm"
  OPTIMIZER: "SGD"
  MULTI_TASK_LEARNING:
    MTL: True  # use MOO/MTL method
#    SAVE_GRAD: True
    SCALER: "DB" # MOO method name

```

* Determine the hyperparameter configurations that control the optimization process, such as the iteration steps `max_iter` and the weights for different loss values:
```yaml
SOLVER:
  META_SOLVER: "AtomicSolver"
  MAX_ITER: 20
  TARGET_DENSITY: 0.225
  SIGMOID_STEEPNESS: 4.0
```


* Configure the remaining project parameters, such as `OUTPUT_DIR`:
```yaml
OUTPUT_DIR: "exps/simpleilt_2048"
```