# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

import hydra
from omegaconf import DictConfig

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import onnx
import onnxruntime as ort

import datetime
import os
import torch
import numpy as np
from matplotlib import pyplot as plt


class ModelWrapper(torch.nn.Module):
    '''
    Main idea is to ignore outputs which we don't need from model
    '''
    def __init__(self, model):
        torch.nn.Module.__init__(self)
        self._model = model
        
        
    def forward(self,input_dict):
        input_dict['obs'] = self._model.norm_obs(input_dict['obs'])
        '''
        just model export doesn't work. Looks like onnx issue with torch distributions
        thats why we are exporting only neural network
        '''
        #print(input_dict)
        #output_dict = self._model.a2c_network(input_dict)
        #input_dict['is_train'] = False
        #return output_dict['logits'], output_dict['values']
        return self._model.a2c_network(input_dict)

class RLGTrainer():
    def __init__(self, cfg, cfg_dict):
        self.cfg = cfg
        self.cfg_dict = cfg_dict

    def launch_rlg_hydra(self, env):
        # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
        # We use the helper function here to specify the environment config.
        self.cfg_dict["task"]["test"] = self.cfg.test

        # register the rl-games adapter to use inside the runner
        vecenv.register('RLGPU',
                        lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        env_configurations.register('rlgpu', {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: env
        })

        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

    def run(self):
        # create runner and set the settings
        runner = Runner(RLGPUAlgoObserver())
        runner.load(self.rlg_config_dict)
        runner.reset()

        # dump config dict
        experiment_dir = os.path.join('runs', self.cfg.train.params.config.name)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        agent = runner.create_player()
        agent.restore(self.cfg.checkpoint)

        import rl_games.algos_torch.flatten as flatten
        inputs = {
            'obs' : torch.zeros((1,) + agent.obs_shape).to(agent.device),
            'rnn_states' : agent.states,
        }
        with torch.no_grad():
            adapter = flatten.TracingAdapter(ModelWrapper(agent.model), inputs,allow_non_tensor=True)
            traced = torch.jit.trace(adapter, adapter.flattened_inputs,check_trace=False)
            flattened_outputs = traced(*adapter.flattened_inputs)
            print(flattened_outputs)
        
        export_file = "mobilefranka.onnx"

        torch.onnx.export(traced, *adapter.flattened_inputs, export_file, verbose=True, input_names=['obs'], output_names=['mu', 'log_std', 'value'])

        onnx_model = onnx.load(export_file)

        # Check that the model is well formed
        onnx.checker.check_model(onnx_model)

        ort_model = ort.InferenceSession(export_file)

        outputs = ort_model.run(
            None,
            {"obs": np.zeros((1,)+agent.obs_shape).astype(np.float32)},
        )
        #print(outputs)
        #action = np.argmax(outputs[0])
        #print(action)

        #input()

        is_done = False
        env = agent.env
        obs = env.reset()
        #obs, reward, done, info = env.step(torch.tensor([[0.0]]))
        #print(obs)

        obs = env.reset()
        #obs, reward, done, info = env.step(torch.tensor([[0.0]]))
        #print(obs)
        
        #input()
        #input()
        #prev_screen = env.render(mode='rgb_array')
        #plt.imshow(prev_screen)
        total_reward = 0
        num_steps = 0
        while not is_done:
            #obs["obs"][:, 2] = 6.254

            # add some noise to the pose observations to check if optitrack noise affects the performance
            obs["obs"][:, :3] += torch.rand(3).to(agent.device) * 0.05
            obs["obs"][:, 21:24] += torch.rand(3).to(agent.device) * 0.05
            #obs["obs"][:, :3] = torch.zeros(3).to(agent.device)
            #obs["obs"][:, 21:24] += torch.tensor([0.76, -0.09, 0.85]).to(agent.device)

            # set the target location manually to test
            # obs["obs"][:, -5:-2] = torch.tensor([2.0, 1.0, 0.5]).to(agent.device)
            
            outputs = ort_model.run(None, {"obs": obs["obs"].cpu().numpy()},)
            #print("outputs[0]", outputs[0])
            mu = outputs[0] #.squeeze(0)
            sigma = np.exp(outputs[1]) #.squeeze(0))
            #action = np.random.normal(mu, sigma)
            action = mu
            action = np.clip(action, -1.0, 1.0)
            action = torch.tensor(action)
            #action = torch.tensor(mu)
            #print(mu, sigma, action)
            #print(action)
            #print(obs)
            #self.polar_to_cartesian_coordinate(obs["obs"].cpu().numpy().squeeze()[:36], -np.pi, 0)
            #input()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            num_steps += 1
            is_done = done[0]
            #print("onnx,obs", obs)
            #screen = env.render(mode='rgb_array')
            #plt.imshow(screen)
            #display.display(plt.gcf())
            #display.clear_output(wait=True)
        print(total_reward, num_steps)
        #ipythondisplay.clear_output(wait=True)

        # runner.run({
        #     'train': not self.cfg.test,
        #     'play': self.cfg.test,
        #     'checkpoint': self.cfg.checkpoint,
        #     'sigma': None
        # })
    
    def polar_to_cartesian_coordinate(self, ranges, angle_min, angle_max):
        angle_step = (angle_max - angle_min) / len(ranges)
        angle = 180
        points = []
        for range in ranges:
            x = range * np.cos(angle)
            y = range * np.sin(angle)
            angle += angle_step
            points.append([x,y])

        points_np = np.array(points)
        #print(points_np)
        plt.figure()
        colors = np.linspace(0, 1, 36)
        sizes = np.linspace(1, 20, 36)
        plt.scatter(points_np[:,0], points_np[:,1], c=colors, s=sizes)
        plt.show()
    
        return points


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    headless = cfg.headless
    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    if cfg_dict["test"]:
        cfg_dict["task"]["env"]["numEnvs"] = 1
        cfg_dict["train"]["params"]["config"]["minibatch_size"] = cfg_dict["train"]["params"]["config"]["horizon_length"]
        #cfg_dict["task"]["domain_randomization"]["randomize"] = False

    task = initialize_task(cfg_dict, env)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    #cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg.seed = set_seed(-1, torch_deterministic=cfg.torch_deterministic)


    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    rlg_trainer.run()
    env.close()


if __name__ == '__main__':
    parse_hydra_configs()
