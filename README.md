# CMLO: Constrained-Model-shift-Lower-bound-Optimization

Code for [When to Update Your Model: Constrained Model-based Reinforcement Learning](https://arxiv.org/abs/2210.08349)


## Run the Code:

Depending on what interface you want to use to run this code, there are three main ways you can run the code:
### Run CMLO under Cat-Runner 

1. according to readme.md, start cat-runner, that is, you can open a local web page, direct interaction can be, different parameters according to the yaml file to modify on the line

2. still start cat-runner, but do not need to open the web page. Then open the terminal, you can use the following command:
`tools/run`: start a new task.
`tools/stop`: stop a running task.
`tools/add_to_tensorboard`: add a task to the tensorboard list
`tools/del_from_tensorboard`: removes a task from the tensorboard list.

> Notes: All commands come with their own parameter descriptions and can be called without parameters.
Open the website and check the status of Cat Runner.
In the task list, the "Status" column can be clicked to see the detailed parameters and output files.

### Run CMLO without Cat-Runner 

3. Without starting Cat Runner, just run it from the command line as:
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

