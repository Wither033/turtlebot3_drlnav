#!/usr/bin/env python3
#
# Parallel training helpers for turtlebot3_drlnav.

import json
import os
import socket
import tempfile
import time
from pathlib import Path


DEFAULT_TRANSITION_HOST = "127.0.0.1"
DEFAULT_TRANSITION_PORT = 46000


def ensure_stage_file(stage):
    """Keep compatibility with upstream modules that read the stage from /tmp."""
    with open("/tmp/drlnav_current_stage.txt", "w", encoding="utf-8") as stage_file:
        stage_file.write(str(int(stage)))


def base_path():
    return os.getenv("DRLNAV_BASE_PATH", "/home/turtlebot3_drlnav")


def default_weight_dir():
    return os.path.join(base_path(), "src", "turtlebot3_drl", "model", "_parallel_policy")


def create_model(algorithm, device, sim_speed):
    from .dqn import DQN
    from .ddpg import DDPG
    from .td3 import TD3

    if algorithm == "dqn":
        return DQN(device, sim_speed)
    if algorithm == "ddpg":
        return DDPG(device, sim_speed)
    if algorithm == "td3":
        return TD3(device, sim_speed)
    raise ValueError("invalid algorithm specified (%s), choose one of: dqn, ddpg, td3" % algorithm)


def to_jsonable(value):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


class JsonLineClient:
    def __init__(self, host, port, retry_seconds=2.0):
        self.host = host
        self.port = int(port)
        self.retry_seconds = float(retry_seconds)
        self.sock = None

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def connect(self):
        while self.sock is None:
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=10)
            except OSError as exc:
                print("waiting for learner at %s:%s (%s)" % (self.host, self.port, exc))
                time.sleep(self.retry_seconds)

    def send(self, payload):
        line = json.dumps(to_jsonable(payload), separators=(",", ":")) + "\n"
        encoded = line.encode("utf-8")
        while True:
            self.connect()
            try:
                self.sock.sendall(encoded)
                return
            except OSError:
                self.close()
                time.sleep(self.retry_seconds)


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(to_jsonable(payload), tmp_file, indent=2, sort_keys=True)
            tmp_file.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def publish_actor_policy(model, policy_dir, metadata):
    import torch

    policy_dir = Path(policy_dir)
    policy_dir.mkdir(parents=True, exist_ok=True)
    actor_path = policy_dir / "actor_latest.pt"
    tmp_path = policy_dir / "actor_latest.pt.tmp"
    torch.save(model.actor.state_dict(), tmp_path)
    os.replace(tmp_path, actor_path)

    payload = dict(metadata)
    payload["actor_path"] = str(actor_path)
    atomic_write_json(policy_dir / "policy_meta.json", payload)


def load_actor_policy_if_new(model, policy_dir, current_version, device):
    import torch

    meta_path = Path(policy_dir) / "policy_meta.json"
    if not meta_path.exists():
        return current_version, None

    metadata = read_json(meta_path)
    version = int(metadata.get("version", -1))
    if version <= current_version:
        return current_version, metadata

    actor_path = metadata.get("actor_path")
    if not actor_path or not os.path.exists(actor_path):
        return current_version, metadata

    state_dict = torch.load(actor_path, map_location=device)
    model.actor.load_state_dict(state_dict)
    if hasattr(model, "epsilon") and "epsilon" in metadata:
        model.epsilon = float(metadata["epsilon"])
    print("loaded policy version %s from %s" % (version, actor_path))
    return version, metadata
