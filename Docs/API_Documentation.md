# API Documentation

## AtomicSolver [source](../mtilt/solving/solver/solver.py#L13)
The most basic atomic solver.
Receives a design instance passed by the Trainer as input, and pairs it with pre-defined solving modules to optimize the instance, such as objectives and lithography models. It then internally solves the task and returns the optimized mask obtained.

**<font color="grey" size=5>Method: \__call\__(self, inputs, algo, objectives, model, iters):</font>**

  * Parameters: 
    * inputs(*dict*) - A dictionary containing three key-value pairs of type (`str`, `torch.Tensor`): backup serves as the detached leaf node for optimizing the mask and constructs the computation graph, params represents the mask to be optimized, and target holds the current lithographic target mask.   
    * algo(*class*) - A configurable class for the optimization process. It takes the `losses` as inputs and updates the corresponding mask.
    * objectives(*class*) - A configurable class to customize objectives. It takes the results obtained from the `lithography` model and computes the corresponding loss values based on user-defined loss functions.
    * model(*torch.nn.Module*) - A configurable class to define the lithographic model. It takes `backup` and `params` from the `inputs` and returns the lithographic result..
    * iters(*int*) - The number of iterations for solving each mask internally, with a default of 20.

### Examples:


```python
# define objectives
objectives = build_objectives(cfg)

# define lithographic model
litho_model = build_litho(cfg)

# def initializer
initializer = build_initializer(cfg)

# define solver
solver = build_solver(cfg)

# optional: define mtl scaler
mtl_scaler = build_scaler(cfg)

# define optimization algorithm
algorithm = build_algorithm(cfg, mtl_scaler)

# define iters
max_iter = 20

# get inputs
inputs = next(_data_loader_iter)[0]

# call the solver, get the optimized mask
loss_dict, bestMask, bestParams, l2Min, pvbMin = \
        solver(inputs, algorithm, objectives, model, max_iter)
```



(more modules, coming soon...)