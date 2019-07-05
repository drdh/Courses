*介绍ray中至关重要的tune库*

## 前言

首先回顾一下关于整个ray系列的结构。

- ***Ray: Flexible, High-Performance Distributed Execution Framework***

  *ray作为底层，实现分布式计算的底层框架*

- ***Tune: Scalable Hyperparameter Search***

  *实现机器学习中的超参搜索*

- ***RLlib: Scalable Reinforcement Learning***

  *实现可拓展的强化学习框架*

## Features

- 支持任何深度学习框架，包括PyTorch, TensorFlow, Keras
- 可从一些可拓展的hyperparameter与model搜索技术中选择，比如：
  - [Population Based Training (PBT)](https://ray.readthedocs.io/en/latest/tune-schedulers.html#population-based-training-pbt)
  - [Median Stopping Rule](https://ray.readthedocs.io/en/latest/tune-schedulers.html#median-stopping-rule)
  - [HyperBand](https://ray.readthedocs.io/en/latest/tune-schedulers.html#asynchronous-hyperband)
- 混合不同的hyperparameter优化方式，比如[HyperOpt with HyperBand](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/hyperopt_example.py)
- 可以使用多种方式可视化结果，比如： [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard), [parallel coordinates (Plot.ly)](https://plot.ly/python/parallel-coordinates-plot/),  [rllab’s VisKit](https://media.readthedocs.org/pdf/rllab/latest/rllab.pdf).
- 无需修改代码就可以拓展到在大分布式集群上运行
- 并行化，比如使用GPU或者算法本身就支持并行化和分布式，使用 Tune’s [resource-aware scheduling](https://ray.readthedocs.io/en/latest/tune-usage.html#using-gpus-resource-allocation),

## Introduction

### Overview

![_images/tune-api.svg](Tune/tune-api.svg)

Tune在集群中调度一系列的trials. 每个trials执行一个用户自定义的Python function或者class, 并且每个trial 的参数要么由`config`变体(Tune's Variant Generator)定义, 要么由用户自定义的搜索算法定义。最后，所有的trials由trial scheduler定义

### Experiment 流程

####  构建model

如下使用keras构建一个识别mnist数据中数字的CNN

是一个很平常的神经网络，返回的model就是输出image然后输出分类0-9

```python
def make_model(parameters):
    config = DEFAULT_ARGS.copy()  # This is obtained via the global scope
    config.update(parameters)
    num_classes = 10
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(config["kernel1"], config["kernel1"]),
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (config["kernel2"], config["kernel2"]), activation='relu'))
    model.add(MaxPooling2D(pool_size=(config["poolsize"], config["poolsize"])))
    model.add(Dropout(config["dropout1"]))
    model.add(Flatten())
    model.add(Dense(config["hidden"], activation='relu'))
    model.add(Dropout(config["dropout2"]))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(
                      lr=config["lr"], momentum=config["momentum"]),
                  metrics=['accuracy'])
    return model
```

#### 使用tune来训练模型

主要是寻找超参

两种方式实现training的代码构建

- Function

  这里定义的函数需要传入两个对象，config和reporter, 其中config是用于超参搜索的dict内容，reporter用于返回metrics给Tune来控制什么时候停止

  ```python
  def train_mnist_tune(config, reporter):
      data_generator = load_data()
      model = make_model(config)
      for i, (x_batch, y_batch) in enumerate(data_generator):
          model.fit(x_batch, y_batch, verbose=0)
          if i % 3 == 0:
              last_checkpoint = "weights_tune_{}.h5".format(i)
              model.save_weights(last_checkpoint)
          mean_accuracy = model.evaluate(x_batch, y_batch)[1]
          reporter(mean_accuracy=result[1], timesteps_total=i, checkpoint=last_checkpoint)
  ```

- Class

  作为`ray.tune.Trainable`的子类。RLlib中的优化相关类就是作为`Trainable`的子类

#### 设定超参的搜索空间以及其他

两种方式设置运行的config

- Python
- JSON

传入的参数重要的有

> `stop(dict)`: 确什么时候停止，比如设定准确率为0.95
>
> `config`: 设定超参的搜索空间，比如Learning Rate设为一系列小数

下面是使用Python的一个案例，使用Json很类似。

```python
configuration = tune.Experiment(
    "experiment_name",
    run=train_mnist_tune,
    trial_resources={"cpu": 4},
    stop={"mean_accuracy": 0.95},  
    config={"lr": lambda spec: np.random.uniform(0.001, 0.1),
            "momentum": tune.grid_search([0.2, 0.4, 0.6])} 
)
```

#### 执行

传入的重要参数有

> 上面的tune.Experiment 一系列设定
>
> search_alg 优化的搜索算法
>
> scheduler 执行的调度器。下面会设定这一个

```python
trials = tune.run_experiments(configuration, verbose=False)
```

#### 使用调度器

最大的不同之处就在于，使用了`AsyncHyperBandScheduler`调度器。如果不指定调度器，就会直接使用`FIFOScheduler`

主要目的是更好地利用分布的资源

```python
configuration2 = tune.Experiment(
    "experiment2",
    run=train_mnist_tune,
    num_samples=5, 
    trial_resources={"cpu": 4},
    stop={"mean_accuracy": 0.95},
    config={
        "lr": lambda spec: np.random.uniform(0.001, 0.1),
        "momentum": tune.grid_search([0.2, 0.4, 0.6]),
        "hidden": lambda spec: np.random.randint(16, high=513),
    }
)

hyperband = AsyncHyperBandScheduler(
    time_attr='timesteps_total',
    reward_attr='mean_accuracy')
    
trials = tune.run_experiments(configuration2, scheduler=hyperband, verbose=False)
```





