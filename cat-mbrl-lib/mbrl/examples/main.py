# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import omegaconf
import torch
import os


import mbrl.util.env


@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)

    if 'seed_override' in os.environ:
            cfg.seed = int(os.environ['seed_override'])
            print("Using overridden seed %s" % cfg.seed)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        algorithm_name = os.environ['algo'] if 'algo' in os.environ else 'mbpo'
        print("---Current Running Algorithm : %s ---"%algorithm_name)
        if algorithm_name == 'cmlo':
            import mbrl.algorithms.cmlo as cmlo
            cmlo.init(cwd)
            return cmlo.train(env, test_env, term_fn, cfg)


import sys

if __name__ == "__main__":   
    cwd = os.getcwd()
    sys.stdout.reconfigure(line_buffering=True)  #* flush stdout every line
    run()