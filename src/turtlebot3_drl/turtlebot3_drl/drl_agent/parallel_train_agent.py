#!/usr/bin/env python3
#
# Central learner for parallel off-policy training.

import argparse
import json
import os
import queue
import socket
import sys
import threading
import time
from collections import deque

from .parallel_common import (
    DEFAULT_TRANSITION_HOST,
    DEFAULT_TRANSITION_PORT,
    default_weight_dir,
    ensure_stage_file,
    publish_actor_policy,
)


class TransitionServer:
    def __init__(self, host, port, output_queue):
        self.host = host
        self.port = int(port)
        self.output_queue = output_queue
        self.sock = None
        self.thread = None
        self.stopped = threading.Event()

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()
        print("parallel learner listening on %s:%s" % (self.host, self.port))

    def stop(self):
        self.stopped.set()
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass

    def _accept_loop(self):
        while not self.stopped.is_set():
            try:
                conn, address = self.sock.accept()
            except OSError:
                if not self.stopped.is_set():
                    raise
                return
            print("worker connected from %s:%s" % address)
            thread = threading.Thread(target=self._client_loop, args=(conn, address), daemon=True)
            thread.start()

    def _client_loop(self, conn, address):
        try:
            with conn:
                reader = conn.makefile("r", encoding="utf-8")
                for line in reader:
                    if not line.strip():
                        continue
                    try:
                        self.output_queue.put(json.loads(line))
                    except json.JSONDecodeError as exc:
                        print("bad worker message from %s:%s: %s" % (address[0], address[1], exc))
        finally:
            print("worker disconnected from %s:%s" % address)


class ParallelLearner:
    def __init__(self, args):
        self.args = args
        ensure_stage_file(args.stage)
        if args.base_path:
            os.environ["DRLNAV_BASE_PATH"] = args.base_path

        from ..common import utilities as util
        from ..common.logger import Logger
        from ..common.replaybuffer import ReplayBuffer
        from ..common.settings import GRAPH_DRAW_INTERVAL, MODEL_STORE_INTERVAL, OBSERVE_STEPS
        from ..common.storagemanager import StorageManager
        from .parallel_common import create_model

        self.util = util
        self.graph_draw_interval = GRAPH_DRAW_INTERVAL
        self.model_store_interval = MODEL_STORE_INTERVAL
        self.observe_steps = OBSERVE_STEPS

        self.algorithm = args.algorithm
        self.device = util.check_gpu()
        self.sim_speed = util.get_simulation_speed(util.stage)
        self.model = create_model(self.algorithm, self.device, self.sim_speed)
        self.replay_buffer = ReplayBuffer(self.model.buffer_size)

        self.sm = StorageManager(self.algorithm, args.load_session, args.load_episode, self.device, util.stage)
        self.episode = int(args.load_episode)
        self.total_steps = 0
        self.graphdata = [self.total_steps, [], [], [], []]

        if args.load_session:
            del self.model
            self.model = self.sm.load_model()
            self.model.device = self.device
            self.sm.load_weights(self.model.networks)
            self.replay_buffer.buffer = self.sm.load_replay_buffer(
                self.model.buffer_size,
                os.path.join(args.load_session, "stage" + str(self.sm.stage) + "_latest_buffer.pkl"),
            )
            self.graphdata = self.sm.load_graphdata()
            self.total_steps = self.graphdata[0]
            print("loaded model %s at episode %s, global steps %s" % (args.load_session, self.episode, self.total_steps))
        else:
            self.sm.new_session_dir(util.stage)
            self.sm.store_model(self.model)

        self.logger = Logger(
            1,
            self.sm.machine_dir,
            self.sm.session_dir,
            self.sm.session,
            self.model.get_model_parameters(),
            self.model.get_model_configuration(),
            str(util.stage),
            self.algorithm,
            self.episode,
        )

        self.policy_dir = args.weight_dir or default_weight_dir()
        self.policy_version = 0
        self.loss_critic_sum = 0.0
        self.loss_actor_sum = 0.0
        self.loss_update_count = 0

        self.transition_queue = queue.Queue(maxsize=args.queue_size)
        self.server = TransitionServer(args.host, args.port, self.transition_queue)

    def publish_policy(self):
        publish_actor_policy(
            self.model,
            self.policy_dir,
            {
                "algorithm": self.algorithm,
                "epsilon": getattr(self.model, "epsilon", 0.0),
                "observe_steps": self.observe_steps,
                "policy_dir": self.policy_dir,
                "session": self.sm.session,
                "stage": int(self.args.stage),
                "total_steps": self.total_steps,
                "version": self.policy_version,
            },
        )
        print("published policy version %s at global step %s" % (self.policy_version, self.total_steps))

    def train_from_transition(self, msg):
        self.replay_buffer.add_sample(
            msg["state"],
            msg["action"],
            [msg["reward"]],
            msg["next_state"],
            [msg["done"]],
        )
        self.total_steps += 1

        if self.replay_buffer.get_length() >= self.model.batch_size:
            for _ in range(self.args.updates_per_transition):
                loss_c, loss_a = self.model._train(self.replay_buffer)
                self.loss_critic_sum += float(loss_c)
                self.loss_actor_sum += float(loss_a)
                self.loss_update_count += 1

        if self.total_steps % self.args.weight_sync_interval == 0:
            self.policy_version += 1
            self.publish_policy()

        if msg.get("done") and msg.get("episode"):
            self.finish_episode(msg["episode"])

    def finish_episode(self, episode_msg):
        if self.total_steps < self.observe_steps:
            print("Observe phase: %s/%s steps" % (self.total_steps, self.observe_steps))
            return

        self.episode += 1
        step = int(episode_msg["steps"])
        reward_sum = float(episode_msg["reward_sum"])
        outcome = int(episode_msg["outcome"])
        duration = float(episode_msg["duration"])
        env_id = episode_msg.get("env_id", "?")

        if self.loss_update_count > 0:
            avg_loss_critic = self.loss_critic_sum / self.loss_update_count
            avg_loss_actor = self.loss_actor_sum / self.loss_update_count
        else:
            avg_loss_critic = 0.0
            avg_loss_actor = 0.0

        self.graphdata[0] = self.total_steps
        self.graphdata[1].append(outcome)
        self.graphdata[2].append(reward_sum)
        self.graphdata[3].append(avg_loss_critic)
        self.graphdata[4].append(avg_loss_actor)

        print(
            "Epi: %-5s env: %-3s R: %-8.0f outcome: %-13s steps: %-6s steps_total: %-7s time: %-6.2f"
            % (
                self.episode,
                env_id,
                reward_sum,
                self.util.translate_outcome(outcome),
                step,
                self.total_steps,
                duration,
            )
        )

        self.logger.file_log.write(
            "%s, %s, %s, %s, %s, %s, %s, %s, %s\n"
            % (
                self.episode,
                reward_sum,
                outcome,
                duration,
                step,
                self.total_steps,
                self.replay_buffer.get_length(),
                avg_loss_critic,
                avg_loss_actor,
            )
        )
        self.logger.file_log.flush()

        self.loss_critic_sum = 0.0
        self.loss_actor_sum = 0.0
        self.loss_update_count = 0

        should_store = (self.episode % self.model_store_interval == 0) or (self.episode == 1)
        should_publish = (self.episode % self.args.weight_sync_episodes == 0) or (self.episode == 1)
        if should_store:
            self.sm.save_session(self.episode, self.model.networks, self.graphdata, self.replay_buffer.buffer)
        if should_publish:
            self.policy_version += 1
            self.publish_policy()

    def run(self):
        self.server.start()
        self.publish_policy()
        last_print = time.time()
        try:
            while True:
                try:
                    msg = self.transition_queue.get(timeout=1.0)
                except queue.Empty:
                    if time.time() - last_print >= self.args.idle_print_seconds:
                        print(
                            "waiting for samples: queue=%s steps=%s replay=%s episode=%s"
                            % (
                                self.transition_queue.qsize(),
                                self.total_steps,
                                self.replay_buffer.get_length(),
                                self.episode,
                            )
                        )
                        last_print = time.time()
                    continue

                if msg.get("type") != "transition":
                    print("ignoring unknown message: %s" % msg.get("type"))
                    continue
                self.train_from_transition(msg)
        except KeyboardInterrupt:
            print("stopping parallel learner")
        finally:
            self.policy_version += 1
            self.publish_policy()
            self.sm.save_session(self.episode, self.model.networks, self.graphdata, self.replay_buffer.buffer)
            self.server.stop()


def build_parser():
    parser = argparse.ArgumentParser(description="Central learner for parallel turtlebot3_drlnav training.")
    parser.add_argument("algorithm", choices=["dqn", "ddpg", "td3"])
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--host", default=DEFAULT_TRANSITION_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_TRANSITION_PORT)
    parser.add_argument("--base-path", default="")
    parser.add_argument("--weight-dir", default="")
    parser.add_argument("--load-session", default="")
    parser.add_argument("--load-episode", type=int, default=0)
    parser.add_argument("--updates-per-transition", type=int, default=1)
    parser.add_argument("--weight-sync-interval", type=int, default=1000)
    parser.add_argument("--weight-sync-episodes", type=int, default=10)
    parser.add_argument("--queue-size", type=int, default=10000)
    parser.add_argument("--idle-print-seconds", type=int, default=30)
    return parser


def main(argv=sys.argv[1:]):
    args = build_parser().parse_args(argv)
    learner = ParallelLearner(args)
    learner.run()


if __name__ == "__main__":
    main()
