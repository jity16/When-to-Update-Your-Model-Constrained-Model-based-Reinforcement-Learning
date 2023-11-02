# When to Update Your Model: Constrained Model-based Reinforcement Learning

This repository contains the code to reproduce the experimental results of CMLO algorithm in the paper
[**When to Update Your Model: Constrained Model-based Reinforcement Learning**](https://arxiv.org/abs/2210.08349). 



## Overview
Our paper proposes a general scheme for the monotonicity of MBRL methods considering the often-neglected issue of model shifts. The algorithm CMLO (Constrained Model-shift Lower-bound Optimization), stemming from the follow-up constrained optimization problem, has asymptotic performance rivaling the best model-free algorithms and boasts better monotonicity. CMLO introduces an event-triggered mechanism that ﬂexibly determines when to update the model.
Brief overview and more information refer to [Project website](https://jity16.github.io/cmlo/).




## Getting Started
Depending on your preferred interface for running this code, there are two primary methods available:

* [Run CMLO under Cat-Runner](run-cmlo-under-cat-runner): We offer **Cat-Runner**, a lightweight, interactive training task manager that allows you to run and manage tasks without the need for writing code. You can also use the command-line interface to run and manage your CMLO tasks.

* [Quick Run CMLO without Cat-Runner](quick-run-cmlo-without-cat-runner): If you prefer not to set up Cat-Runner, we also provide a quick and easy way to run tasks without it.

### Prerequisites

Install `mbrl` and packages listed in the requirements folder by:
```bash
pip install -e ".[dev]"
```

### Quick Run CMLO without Cat-Runner 
If you prefer not to use Cat-Runner, you can run the code from the command line directly. Use the following command: 
```bash
   cd cat-mbrl-lib/ &&
      CUDA_VISIBLE_DEVICES=%s // gpu_id
      des=%s  // run_id
      algo=%s  //algo_name
      freq=fixed // freq_name
      seed_override=%s  // seed
      python -m mbrl.examples.main
      algorithm=mbpo
      overrides=%s // hyperparamters
```

Here is an example:

```bash
cd cat-mbrl-lib/

CUDA_VISIBLE_DEVICES=0 seed_override=0 des=test algo=cmlo freq=fixed python -m mbrl.examples.main algorithm=mbpo overrides=mbpo_halfcheetah
```



### Run CMLO under Cat-Runner 

**How to Start Cat-Runner Platform**

1. Prerequisites (Cat-Runner only runs on Linux)
   * Install `gcc` and `nodejs`. 
   * Compile the following helper programs.
      ```bash
      gcc tools/do_run.c -O2 -o tools/do_run.exe
      gcc cat-runner/c/check_running.c -O2 -o cat-runner/c/check_running.exe
      gcc cat-runner/c/run.c -O2 -o cat-runner/c/run.exe
      ```
   * Install node dependencies.
      ```bash
      cd cat-runner
      npm install
      ```

2. Start Cat-Runner Server
   * Run `start_server.sh` in `cat-runner` directory. The server will bind on `127.0.0.1:20001`.

   * To choose different binding address and port, invoke
      ```bash
      node index.js <address> <port>
      ```

3. Start Tensorboard Server

   * Run `start_tensorboard.sh` to start tensorboard server. The server will bind on `127.0.0.1:20002`.

   * To choose different binding address and port, invoke
      ```bash
      tensorboard --logdir tensorboard_logdir/ --host <address> --port <port>
      ```

      > Note: There are links to the tensorboard server on Cat-Runner pages. Changing the binding address or port may break these links.



**How to Run CMLO under Cat-Runner**
* Cat-Runner Command-Line Interface
   * Initiate Cat-Runner, but without opening the webpage. Instead, use the terminal to control it with the following scripts:
      * `tools/run`: Start a new task.
      * `tools/stop`: Stop a running task.
      * `tools/add_to_tensorboard`: Add a task to the Tensorboard list
      * `tools/del_from_tensorboard`: Remove a task from the Tensorboard list.

* Cat-Runner Webpage Interface
   * Initiate Cat-Runner, open the webpage, then you can iteract with the webpage to start your training tasks.



## Citation
If you find our work useful, please consider citing the paper as follows:
```
@inproceedings{cmlo,
  title={When to Update Your Model: Constrained Model-based Reinforcement Learning},
  author={Ji, Tianying and Luo, Yu and Sun, Fuchun and Jing, Mingxuan and He, Fengxiang and Huang, Wenbing},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
