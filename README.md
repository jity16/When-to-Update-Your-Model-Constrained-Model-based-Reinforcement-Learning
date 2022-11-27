# CMLO: Constrained-Model-shift-Lower-bound-Optimization

Code for [When to Update Your Model: Constrained Model-based Reinforcement Learning](https://arxiv.org/abs/2210.08349)


## CMLO run under Cat-Runner 
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

