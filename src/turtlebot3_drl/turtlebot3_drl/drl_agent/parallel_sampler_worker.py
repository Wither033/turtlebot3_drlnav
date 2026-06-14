#!/usr/bin/env python3
#
# Environment sampler for parallel off-policy training.

import argparse
import copy
import os
import sys
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import DrlStep, Goal

from .parallel_common import (
    DEFAULT_TRANSITION_HOST,
    DEFAULT_TRANSITION_PORT,
    JsonLineClient,
    default_weight_dir,
    ensure_stage_file,
    load_actor_policy_if_new,
)


class ParallelSamplerWorker(Node):
    def __init__(self, args):
        super().__init__("%s_parallel_sampler_%s" % (args.algorithm, args.env_id))
        self.args = args
        ensure_stage_file(args.stage)
        if args.base_path:
            os.environ["DRLNAV_BASE_PATH"] = args.base_path

        from ..common import utilities as util
        from ..common.settings import ENABLE_STACKING, ENABLE_VISUAL, OBSERVE_STEPS
        from .parallel_common import create_model

        self.rclpy = rclpy
        self.Empty = Empty
        self.DrlStep = DrlStep
        self.Goal = Goal
        self.util = util
        self.enable_stacking = ENABLE_STACKING
        self.enable_visual = ENABLE_VISUAL
        self.observe_steps = OBSERVE_STEPS

        self.algorithm = args.algorithm
        self.env_id = args.env_id
        self.device = util.check_gpu()
        self.sim_speed = util.get_simulation_speed(util.stage)
        self.model = create_model(self.algorithm, self.device, self.sim_speed)

        self.step_comm_client = self.create_client(DrlStep, "step_comm")
        self.goal_comm_client = self.create_client(Goal, "goal_comm")
        self.gazebo_pause = self.create_client(Empty, "/pause_physics")
        self.gazebo_unpause = self.create_client(Empty, "/unpause_physics")

        self.transition_client = JsonLineClient(args.learner_host, args.learner_port)
        self.policy_dir = args.weight_dir or default_weight_dir()
        self.policy_version = -1
        self.policy_meta = None
        self.worker_episode = 0

    def spin_once(self):
        self.rclpy.spin_once(self)

    def refresh_policy(self, force=False):
        if not force and self.policy_meta is not None:
            if self.local_step % self.args.policy_poll_steps != 0:
                return
        self.policy_version, metadata = load_actor_policy_if_new(
            self.model,
            self.policy_dir,
            self.policy_version,
            self.device,
        )
        if metadata is not None:
            self.policy_meta = metadata

    def use_random_action(self):
        if self.policy_meta is None:
            return True
        return int(self.policy_meta.get("total_steps", 0)) < int(self.policy_meta.get("observe_steps", self.observe_steps))

    def choose_action(self, state, step):
        self.refresh_policy()
        if self.use_random_action():
            return self.model.get_action_random()
        if self.policy_version < 0:
            self.refresh_policy(force=True)
        return self.model.get_action(state, True, step, self.enable_visual)

    def send_transition(self, state, action, reward, next_state, done, outcome, distance_traveled, episode_msg=None):
        msg = {
            "type": "transition",
            "env_id": self.env_id,
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": bool(done),
            "outcome": int(outcome),
            "distance_traveled": distance_traveled,
        }
        if episode_msg is not None:
            msg["episode"] = episode_msg
        self.transition_client.send(msg)

    def process(self):
        self.util.pause_simulation(self, False)
        while True:
            self.util.wait_new_goal(self)
            self.refresh_policy(force=True)

            episode_done = False
            step = 0
            reward_sum = 0.0
            action_past = [0.0, 0.0]
            state = self.util.init_episode(self)
            self.local_step = 0

            if self.enable_stacking:
                frame_buffer = [0.0] * (self.model.state_size * self.model.stack_depth * self.model.frame_skip)
                state = [0.0] * (self.model.state_size * (self.model.stack_depth - 1)) + list(state)

            self.util.unpause_simulation(self, False)
            time.sleep(0.5)
            episode_start = time.perf_counter()

            while not episode_done:
                self.local_step = step
                action = self.choose_action(state, step)
                action_current = action
                if self.algorithm == "dqn":
                    action_current = self.model.possible_actions[action]

                next_state, reward, episode_done, outcome, distance_traveled = self.util.step(
                    self,
                    action_current,
                    action_past,
                )
                action_past = copy.deepcopy(action_current)
                reward_sum += reward

                if self.enable_stacking:
                    frame_buffer = frame_buffer[self.model.state_size :] + list(next_state)
                    stacked_next_state = []
                    for depth in range(self.model.stack_depth):
                        start = self.model.state_size * (self.model.frame_skip - 1) + (
                            self.model.state_size * self.model.frame_skip * depth
                        )
                        stacked_next_state += frame_buffer[start : start + self.model.state_size]
                    next_state = stacked_next_state

                episode_msg = None
                if episode_done:
                    duration = time.perf_counter() - episode_start
                    self.worker_episode += 1
                    episode_msg = {
                        "env_id": self.env_id,
                        "worker_episode": self.worker_episode,
                        "steps": step + 1,
                        "reward_sum": reward_sum,
                        "outcome": int(outcome),
                        "duration": duration,
                        "distance_traveled": distance_traveled,
                    }

                self.send_transition(
                    state,
                    action,
                    reward,
                    next_state,
                    episode_done,
                    outcome,
                    distance_traveled,
                    episode_msg,
                )

                state = copy.deepcopy(next_state)
                step += 1
                time.sleep(self.model.step_time)

            self.util.pause_simulation(self, False)
            print(
                "env %s episode %s finished: reward %.0f, outcome %s, steps %s"
                % (self.env_id, self.worker_episode, reward_sum, self.util.translate_outcome(outcome), step)
            )

    def destroy(self):
        self.transition_client.close()
        self.destroy_node()


def build_parser():
    parser = argparse.ArgumentParser(description="Sampler worker for parallel turtlebot3_drlnav training.")
    parser.add_argument("algorithm", choices=["dqn", "ddpg", "td3"])
    parser.add_argument("--env-id", default=os.getenv("DRLNAV_ENV_ID", "0"))
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--learner-host", default=DEFAULT_TRANSITION_HOST)
    parser.add_argument("--learner-port", type=int, default=DEFAULT_TRANSITION_PORT)
    parser.add_argument("--base-path", default="")
    parser.add_argument("--weight-dir", default="")
    parser.add_argument("--policy-poll-steps", type=int, default=25)
    return parser


def main(argv=sys.argv[1:]):
    args, ros_args = build_parser().parse_known_args(argv)

    rclpy.init(args=ros_args)
    worker = ParallelSamplerWorker(args)
    try:
        worker.process()
    finally:
        worker.destroy()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
