# Parallel sampler training

This branch keeps the original `train_agent` path intact and adds a separate
parallel off-policy training path:

- `parallel_train_agent`: one central learner that owns the replay buffer,
  model updates, logging, and checkpoint saving.
- `parallel_sampler_worker`: one sampler per Gazebo environment. Each sampler
  interacts with its own `environment` node and sends transitions to the
  central learner.

The design is intentionally conservative. It does not put multiple TurtleBot3
robots into one Gazebo world. Instead, run multiple isolated Gazebo/ROS2
instances and let them feed one learner.

## Architecture

```text
Gazebo/env 0 -> parallel_sampler_worker 0 \
Gazebo/env 1 -> parallel_sampler_worker 1  \
Gazebo/env 2 -> parallel_sampler_worker 2   -> parallel_train_agent -> model checkpoints
Gazebo/env 3 -> parallel_sampler_worker 3  /
```

Communication:

- Workers send transitions to the learner over TCP JSON lines.
- The learner writes actor weights to a shared `parallel_policy` directory.
- Workers poll that directory and update their local actor policy.
- All workers use random actions until the learner reports that the global
  observe phase is finished.

The original single-environment commands still work:

```bash
ros2 run turtlebot3_drl train_agent ddpg
ros2 run turtlebot3_drl test_agent td3 examples/td3_0 7400
```

## Rebuild after pulling these changes

Inside the Docker container:

```bash
cd /home/turtlebot3_drlnav
colcon build
source install/setup.bash
```

## Start the central learner

Run one learner. This process does not need Gazebo, but it does need the same
workspace mount so model checkpoints and policy files are shared with workers.

```bash
source ~/.bashrc
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 run turtlebot3_drl parallel_train_agent td3 \
  --stage 4 \
  --host 0.0.0.0 \
  --port 46000
```

Useful learner options:

```bash
--updates-per-transition 1      # gradient updates per incoming transition
--weight-sync-interval 1000     # publish actor weights every N transitions
--weight-sync-episodes 10       # publish actor weights every N completed episodes
--load-session td3_0            # resume from an existing session
--load-episode 500
```

The learner saves under the usual model directory:

```text
src/turtlebot3_drl/model/<learner-hostname>/<algorithm>_<n>_stage_<stage>/
```

By default, shared policy files are written under:

```text
src/turtlebot3_drl/model/_parallel_policy/
```

Pass `--weight-dir <path>` to both learner and workers if you want a custom
shared policy directory. Use a custom directory if you run more than one
parallel learner at the same time.

## Start each environment worker

Each worker needs its own Docker container, ROS domain, and Gazebo master URI.
Do not run multiple workers against the same `environment` node.

Example for worker 0:

```bash
docker run -dit \
  --name drlnav_env_0 \
  --hostname drlnav_env_0 \
  --gpus all \
  --privileged \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  -v /path/to/turtlebot3_drlnav:/home/turtlebot3_drlnav \
  --network host \
  turtlebot3_drlnav bash
```

Inside worker 0 container, run four terminals or background jobs:

```bash
source ~/.bashrc
export ROS_DOMAIN_ID=21
export GAZEBO_MASTER_URI=http://127.0.0.1:11345
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py
```

```bash
source ~/.bashrc
export ROS_DOMAIN_ID=21
export GAZEBO_MASTER_URI=http://127.0.0.1:11345
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 run turtlebot3_drl gazebo_goals
```

```bash
source ~/.bashrc
export ROS_DOMAIN_ID=21
export GAZEBO_MASTER_URI=http://127.0.0.1:11345
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 run turtlebot3_drl environment
```

```bash
source ~/.bashrc
export ROS_DOMAIN_ID=21
export GAZEBO_MASTER_URI=http://127.0.0.1:11345
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 run turtlebot3_drl parallel_sampler_worker td3 \
  --stage 4 \
  --env-id 0 \
  --learner-host 127.0.0.1 \
  --learner-port 46000
```

For worker 1, use different isolation values:

```bash
export ROS_DOMAIN_ID=22
export GAZEBO_MASTER_URI=http://127.0.0.1:11346

ros2 run turtlebot3_drl parallel_sampler_worker td3 \
  --stage 4 \
  --env-id 1 \
  --learner-host 127.0.0.1 \
  --learner-port 46000
```

Repeat with unique values for each worker:

```text
worker 0: ROS_DOMAIN_ID=21, GAZEBO_MASTER_URI=http://127.0.0.1:11345
worker 1: ROS_DOMAIN_ID=22, GAZEBO_MASTER_URI=http://127.0.0.1:11346
worker 2: ROS_DOMAIN_ID=23, GAZEBO_MASTER_URI=http://127.0.0.1:11347
worker 3: ROS_DOMAIN_ID=24, GAZEBO_MASTER_URI=http://127.0.0.1:11348
```

Important: the Dockerfile appends `export ROS_DOMAIN_ID=1` to `~/.bashrc`.
Export the worker-specific `ROS_DOMAIN_ID` after `source ~/.bashrc`, as shown
above.

## Background launch example

For quick experiments, start each worker container's four jobs in the
background:

```bash
source ~/.bashrc
export ROS_DOMAIN_ID=21
export GAZEBO_MASTER_URI=http://127.0.0.1:11345
cd /home/turtlebot3_drlnav
source install/setup.bash
mkdir -p parallel_logs

nohup ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py > parallel_logs/0_gazebo.log 2>&1 &
sleep 20
nohup ros2 run turtlebot3_drl gazebo_goals > parallel_logs/0_goals.log 2>&1 &
sleep 3
nohup ros2 run turtlebot3_drl environment > parallel_logs/0_environment.log 2>&1 &
sleep 3
nohup ros2 run turtlebot3_drl parallel_sampler_worker td3 --stage 4 --env-id 0 --learner-host 127.0.0.1 --learner-port 46000 > parallel_logs/0_sampler.log 2>&1 &
```

## Practical limits

Start with two workers before trying four or more. Gazebo is usually the
bottleneck, especially on CPU. More workers can reduce wall-clock sample time,
but they also increase simulation load and can make Gazebo unstable if the host
is undersized.

This implementation is for off-policy algorithms in the repo (`dqn`, `ddpg`,
`td3`). It is not a synchronous vectorized environment and it does not implement
on-policy PPO/A2C style rollouts.
