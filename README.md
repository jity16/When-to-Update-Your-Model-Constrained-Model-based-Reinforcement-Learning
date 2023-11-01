# CMLO: Constrained-Model-shift-Lower-bound-Optimization

Code for [When to Update Your Model: Constrained Model-based Reinforcement Learning](https://arxiv.org/abs/2210.08349)


## Run the Code:

Depending on your preferred interface for running this code, there are three primary methods available:
### Run CMLO under Cat-Runner 

[Cat-Runner Webpage Interface]
1. Refer to **How to Start Cat-Runner**, start the Cat-Runner webpage, and then you can interact directly to start your running tasks. If you want different hyperparameters, modify the YAML file in `cat-mbrl-lib/mbrl/examples/conf'

[Cat-Runner Command-Line Interface]
2. Initiate Cat-Runner, but without opening the web page. Instead, use the terminal to control it with the following commands:
`tools/run`: Start a new task.
`tools/stop`: Stop a running task.
`tools/add_to_tensorboard`: Add a task to the Tensorboard list
`tools/del_from_tensorboard`: Remove a task from the Tensorboard list.

> Notes: Each command comes with its own set of parameters and can be executed with or without additional arguments.
You can also view the status of Cat Runner tasks on the website. In the task list, clicking on the "Status" column will provide access to detailed parameters and output files.

### Run CMLO without Cat-Runner 

3. If you prefer not to use Cat-Runner, you can run the code from the command line directly. Use the following command: 
```
   cd cat-mbrl-lib/ && "
    "CUDA_VISIBLE_DEVICES=%s " // gpu_id
    "des=__catrunner_%s__ " // run_id
    "algo=%s " //algo_name
    "freq=custom__%s " // freq_name
    "seed_override=%s " // seed
    "python -m mbrl.examples.main "
    "algorithm=cmlo "
    "overrides=%s " // env_name
    "> ... /run_status/stdout_%s.txt " // run_id
    "2> ... /run_status/stderr_%s.txt " // run_id"
```


## How to Start Cat-Runner 
### Prerequisites

Cat-Runner only runs on Linux.

1. Install `gcc` and `nodejs`. 

2. Compile the following helper programs.

```bash
gcc tools/do_run.c -O2 -o tools/do_run.exe
gcc cat-runner/c/check_running.c -O2 -o cat-runner/c/check_running.exe
gcc cat-runner/c/run.c -O2 -o cat-runner/c/run.exe
```

3. Install node dependencies.

```bash
cd cat-runner
npm install
```

### Start Cat-Runner Server

Run `start_server.sh` in `cat-runner` directory. The server will bind on `127.0.0.1:20001`.

To choose different binding address and port, invoke

```bash
node index.js <address> <port>
```

### Start Tensorboard Server

Run `start_tensorboard.sh` to start tensorboard server. The server will bind on `127.0.0.1:20002`.

To choose different binding address and port, invoke

```bash
tensorboard --logdir tensorboard_logdir/ --host <address> --port <port>
```

Note: There are links to the tensorboard server on Cat-Runner pages. Changing the binding address or port may break these links.

