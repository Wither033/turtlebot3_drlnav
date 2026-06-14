# turtlebot3_drlnav 从环境配置到训练成果查看完整教程

本文档面向 `Wither033/turtlebot3_drlnav` 的 `codex/parallel-sampler-training` 分支，覆盖从 Linux GPU 环境准备、GitHub 下载、Docker 构建、单环境训练、并行采样训练，到模型成果查看和恢复训练的完整流程。

项目地址：

```text
https://github.com/Wither033/turtlebot3_drlnav
```

推荐分支：

```text
codex/parallel-sampler-training
```

## 0. 你最终会得到什么

完成本文档后，你会有：

- 一个可运行的 `turtlebot3_drlnav` Docker 训练环境。
- 原版训练方式：一个 Gazebo 环境训练一个 `ddpg` / `td3` / `dqn` agent。
- 改进训练方式：多个 Gazebo 环境并行采样，一个 learner 统一训练模型。
- 可查看的训练日志、奖励曲线、模型权重和测试命令。

## 1. 推荐机器配置

推荐使用 Linux GPU 云服务器或本地 Linux GPU 工作站。

最低建议：

```text
OS: Ubuntu 20.04 或 Ubuntu 22.04
GPU: NVIDIA GPU, 8GB VRAM 起步
CPU: 8 vCPU 起步
RAM: 16GB 起步
Disk: 80GB 起步
```

更舒服的配置：

```text
GPU: NVIDIA A10 / L4 / RTX 3090 / RTX 4090 / A100
VRAM: 16GB 到 24GB 以上
CPU: 16 vCPU 以上
RAM: 32GB 以上
Disk: 150GB 以上
```

注意：

- 不建议把 Windows 作为正式训练环境。这个项目依赖 Docker、ROS2 Foxy、Gazebo 和 X11 图形显示，Linux 环境更稳。
- Colab Pro 的托管运行时不适合这个项目的正式训练，因为它不适合作为长期 Docker + Gazebo + ROS 多进程训练服务器。
- 如果使用云服务器，建议通过 VS Code Remote-SSH 连接，在远端运行 Docker/Gazebo/训练命令。

## 2. 安装 NVIDIA 驱动并验证 GPU

先确认宿主机能看到 GPU：

```bash
nvidia-smi
```

能看到类似下面的信息即可：

```text
NVIDIA-SMI ... Driver Version ... CUDA Version ...
```

如果 `nvidia-smi` 不存在或报错，先不要继续训练。需要先安装或修复 NVIDIA Linux 驱动。

## 3. 安装 Docker Engine

以下是 Ubuntu 上使用 Docker 官方 apt 仓库的安装方式。Docker 官方文档说明，首次安装前需要先配置 Docker apt repository，然后从该 repository 安装 Docker Engine。

```bash
sudo apt update
sudo apt install -y ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

启动并验证 Docker：

```bash
sudo systemctl enable --now docker
sudo docker run --rm hello-world
```

让当前用户不需要每次都写 `sudo`：

```bash
sudo usermod -aG docker $USER
```

执行后退出 SSH，再重新登录。重新登录后验证：

```bash
docker run --rm hello-world
```

## 4. 安装 NVIDIA Container Toolkit

Docker 本身不能自动把 GPU 透传进容器，需要 NVIDIA Container Toolkit。

安装仓库和软件包：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

配置 Docker 使用 NVIDIA runtime，并重启 Docker：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证容器里能看到 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi
```

如果这一步失败，先不要构建项目镜像。常见原因是：

- 宿主机 NVIDIA 驱动没装好。
- Docker 没重启。
- `nvidia-container-toolkit` 没装好。
- 云厂商镜像没有正确启用 GPU 驱动。

## 5. 准备图形显示

原项目训练需要启动 Gazebo。即使主要训练逻辑是强化学习，Gazebo 仍然是模拟环境，所以需要图形显示能力。

本地 Linux 桌面：

```bash
xhost +local:docker
echo $DISPLAY
```

如果是 SSH 云服务器，有三种常见选择：

```text
方案 A: 使用云桌面 / VNC / NoMachine / NICE DCV
方案 B: 使用 SSH X11 forwarding
方案 C: 使用虚拟显示 Xvfb，仅用于不看 GUI 的训练
```

为了贴近原项目，建议初次训练使用可见的桌面或 VNC。等确认训练链路跑通后，再考虑 Xvfb 或后台训练。

## 6. 从 GitHub 下载项目

选择一个工作目录，例如：

```bash
mkdir -p ~/work
cd ~/work
```

下载你的 GitHub 分支：

```bash
git clone -b codex/parallel-sampler-training https://github.com/Wither033/turtlebot3_drlnav.git
cd turtlebot3_drlnav
```

确认分支：

```bash
git branch --show-current
git log --oneline -3
```

应该看到：

```text
codex/parallel-sampler-training
```

## 7. 构建 Docker 镜像

项目根目录有原版 `Dockerfile`，它会构建：

```text
Ubuntu 20.04
CUDA 11.3.1 base image
Python 3.8
PyTorch 1.10.0+cu113
ROS2 Foxy
Gazebo
TurtleBot3 依赖
```

构建镜像：

```bash
docker build -t turtlebot3_drlnav .
```

第一次构建可能需要 10 到 30 分钟，取决于网络和机器性能。

构建完成后查看镜像：

```bash
docker images | grep turtlebot3_drlnav
```

## 8. 启动训练容器

在项目根目录执行：

```bash
xhost +local:docker
```

启动一个单环境训练容器：

```bash
docker run -it \
  --name drlnav_single \
  --hostname drlnav_single \
  --gpus all \
  --privileged \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env DISPLAY=${DISPLAY} \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/home/turtlebot3_drlnav \
  --network host \
  turtlebot3_drlnav bash
```

如果容器已经创建过，重新进入：

```bash
docker start -ai drlnav_single
```

容器内验证：

```bash
source ~/.bashrc
cd /home/turtlebot3_drlnav
echo $DRLNAV_BASE_PATH
python3 --version
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## 9. 构建 ROS2 工作区

在容器内执行：

```bash
cd /home/turtlebot3_drlnav
colcon build
source install/setup.bash
```

检查包是否可见：

```bash
ros2 pkg list | grep turtlebot3
```

至少应该能看到：

```text
turtlebot3_drl
turtlebot3_gazebo
```

## 10. 原版单环境训练

这是原项目 README 的标准训练方式。需要四个终端，顺序启动。

推荐在容器里用 `tmux`：

```bash
tmux new -s drlnav
```

在 tmux 里按 `Ctrl+b` 再按 `%` 或 `"` 分屏，也可以直接开四个 Docker exec 终端。

### 终端 1: 启动 Gazebo stage

```bash
source ~/.bashrc
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py
```

等 Gazebo 窗口出现，并且机器人、场景加载完成。

### 终端 2: 启动 goal 管理节点

```bash
source ~/.bashrc
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 run turtlebot3_drl gazebo_goals
```

### 终端 3: 启动环境节点

```bash
source ~/.bashrc
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 run turtlebot3_drl environment
```

### 终端 4: 启动训练 agent

DDPG：

```bash
source ~/.bashrc
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 run turtlebot3_drl train_agent ddpg
```

TD3：

```bash
ros2 run turtlebot3_drl train_agent td3
```

DQN：

```bash
ros2 run turtlebot3_drl train_agent dqn
```

推荐先用 `td3` 或 `ddpg`。如果你刚开始，只跑一个 stage4 + td3 或 stage4 + ddpg，确认日志和模型能正常保存。

## 11. 训练参数怎么设置

主要配置文件：

```text
src/turtlebot3_drl/turtlebot3_drl/common/settings.py
```

常用参数：

```python
MODEL_STORE_INTERVAL = 100
GRAPH_DRAW_INTERVAL = 10
EPISODE_TIMEOUT_SECONDS = 50
BATCH_SIZE = 128
BUFFER_SIZE = 1000000
OBSERVE_STEPS = 25000
STEP_TIME = 0.01
```

含义：

```text
MODEL_STORE_INTERVAL: 每多少个 episode 保存一次模型
GRAPH_DRAW_INTERVAL: 每多少个 episode 保存一次训练曲线
EPISODE_TIMEOUT_SECONDS: 单个 episode 最长模拟时间
BATCH_SIZE: 每次训练采样的 batch 大小
BUFFER_SIZE: replay buffer 最大容量
OBSERVE_STEPS: 训练初期随机动作探索步数
STEP_TIME: 每一步后的 sleep 时间，越小训练越快，但 Gazebo 压力越大
```

修改配置后重新构建：

```bash
cd /home/turtlebot3_drlnav
colcon build
source install/setup.bash
```

建议：

- 初次跑通不要改参数。
- 第一次确认模型能保存后，再考虑调小 `MODEL_STORE_INTERVAL`。
- 不要同时改很多参数，否则训练失败时很难判断原因。

## 12. 恢复训练

训练模型保存在：

```text
src/turtlebot3_drl/model/<HOSTNAME>/<MODEL_NAME>/
```

例如：

```text
src/turtlebot3_drl/model/drlnav_single/ddpg_0_stage_4/
```

恢复训练命令格式：

```bash
ros2 run turtlebot3_drl train_agent <algorithm> "<model_name>" <episode>
```

例子：

```bash
ros2 run turtlebot3_drl train_agent ddpg "ddpg_0" 500
```

注意：

- 先启动 Gazebo、`gazebo_goals`、`environment`。
- `<model_name>` 通常是 `ddpg_0`、`td3_0` 这种名字。
- `<episode>` 必须对应已保存的权重 episode。

## 13. 测试已有模型

测试命令格式：

```bash
ros2 run turtlebot3_drl test_agent <algorithm> "<model_name>" <episode>
```

例子：

```bash
ros2 run turtlebot3_drl test_agent td3 "examples/td3_0" 7400
```

测试前也要先启动：

```bash
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py
ros2 run turtlebot3_drl gazebo_goals
ros2 run turtlebot3_drl environment
```

## 14. 并行采样训练

本分支新增了并行采样训练方式。它不是把多个机器人塞进同一个 Gazebo，而是运行多个独立 Gazebo 环境，每个环境一个 sampler worker，所有 worker 把样本发给一个中央 learner。

结构：

```text
Gazebo/env 0 -> parallel_sampler_worker 0 \
Gazebo/env 1 -> parallel_sampler_worker 1  \
Gazebo/env 2 -> parallel_sampler_worker 2   -> parallel_train_agent -> replay buffer -> model
Gazebo/env 3 -> parallel_sampler_worker 3  /
```

适合：

- 想加快 off-policy 算法采样速度。
- 使用 `dqn` / `ddpg` / `td3`。
- 有足够 CPU 和 GPU 资源。

不适合：

- PPO/A2C 这种 on-policy 同步 rollout。
- CPU 很弱的机器。
- 只有一个小 GPU 且 Gazebo 已经卡顿的机器。

### 14.1 先构建一次工作区

任意一个容器内：

```bash
cd /home/turtlebot3_drlnav
colcon build
source install/setup.bash
```

### 14.2 启动 learner 容器

在宿主机项目根目录：

```bash
docker run -dit \
  --name drlnav_learner \
  --hostname drlnav_learner \
  --gpus all \
  --privileged \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  -v $(pwd):/home/turtlebot3_drlnav \
  --network host \
  turtlebot3_drlnav bash
```

进入 learner 容器：

```bash
docker exec -it drlnav_learner bash
```

启动 learner：

```bash
source ~/.bashrc
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 run turtlebot3_drl parallel_train_agent td3 \
  --stage 4 \
  --host 0.0.0.0 \
  --port 46000
```

learner 会做这些事：

```text
接收 worker 发送的 transition
写入 replay buffer
训练 TD3/DDPG/DQN 模型
定期保存 checkpoint
定期发布 actor 权重给 worker
```

### 14.3 启动 worker 容器

每个 worker 都需要独立：

```text
container name
hostname
ROS_DOMAIN_ID
GAZEBO_MASTER_URI
```

推荐先开 2 个 worker，稳定后再加到 4 个。

隔离表：

```text
worker 0: ROS_DOMAIN_ID=21, GAZEBO_MASTER_URI=http://127.0.0.1:11345
worker 1: ROS_DOMAIN_ID=22, GAZEBO_MASTER_URI=http://127.0.0.1:11346
worker 2: ROS_DOMAIN_ID=23, GAZEBO_MASTER_URI=http://127.0.0.1:11347
worker 3: ROS_DOMAIN_ID=24, GAZEBO_MASTER_URI=http://127.0.0.1:11348
```

worker 0 容器：

```bash
docker run -dit \
  --name drlnav_env_0 \
  --hostname drlnav_env_0 \
  --gpus all \
  --privileged \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env DISPLAY=${DISPLAY} \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/home/turtlebot3_drlnav \
  --network host \
  turtlebot3_drlnav bash
```

进入 worker：

```bash
docker exec -it drlnav_env_0 bash
```

在 worker 0 里启动四个进程。

终端 1：

```bash
source ~/.bashrc
export ROS_DOMAIN_ID=21
export GAZEBO_MASTER_URI=http://127.0.0.1:11345
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py
```

终端 2：

```bash
source ~/.bashrc
export ROS_DOMAIN_ID=21
export GAZEBO_MASTER_URI=http://127.0.0.1:11345
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 run turtlebot3_drl gazebo_goals
```

终端 3：

```bash
source ~/.bashrc
export ROS_DOMAIN_ID=21
export GAZEBO_MASTER_URI=http://127.0.0.1:11345
cd /home/turtlebot3_drlnav
source install/setup.bash

ros2 run turtlebot3_drl environment
```

终端 4：

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

worker 1 只需要改隔离参数和 `--env-id`：

```bash
export ROS_DOMAIN_ID=22
export GAZEBO_MASTER_URI=http://127.0.0.1:11346

ros2 run turtlebot3_drl parallel_sampler_worker td3 \
  --stage 4 \
  --env-id 1 \
  --learner-host 127.0.0.1 \
  --learner-port 46000
```

注意：

- Dockerfile 的 `~/.bashrc` 默认写了 `ROS_DOMAIN_ID=1`，所以必须在 `source ~/.bashrc` 后重新 export worker 自己的 `ROS_DOMAIN_ID`。
- worker 和 learner 默认通过 `127.0.0.1:46000` 通信，因为都使用 `--network host`。
- 多个 worker 共享同一个仓库挂载目录，所以都能读到 learner 发布的 actor 权重。

## 15. 后台运行并查看日志

单环境后台启动示例：

```bash
mkdir -p run_logs

nohup ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py > run_logs/gazebo.log 2>&1 &
sleep 20
nohup ros2 run turtlebot3_drl gazebo_goals > run_logs/goals.log 2>&1 &
sleep 3
nohup ros2 run turtlebot3_drl environment > run_logs/environment.log 2>&1 &
sleep 3
nohup ros2 run turtlebot3_drl train_agent td3 > run_logs/train_agent.log 2>&1 &
```

查看日志：

```bash
tail -f run_logs/train_agent.log
```

查看 ROS/Gazebo 进程：

```bash
ps -ef | grep -E 'gazebo|gzserver|gazebo_goals|environment|train_agent|parallel'
```

停止当前训练进程：

```bash
pkill -f 'train_agent|parallel_train_agent|parallel_sampler_worker|environment|gazebo_goals|ros2 launch|gzserver|gzclient'
```

## 16. 成果文件在哪里

训练成果目录：

```text
src/turtlebot3_drl/model/<HOSTNAME>/<MODEL_NAME>/
```

例子：

```text
src/turtlebot3_drl/model/drlnav_learner/td3_0_stage_4/
```

常见文件：

```text
actor_stage4_episode100.pt
target_actor_stage4_episode100.pt
critic_stage4_episode100.pt
target_critic_stage4_episode100.pt
stage4_agent.pkl
stage4_episode100.pkl
stage4_latest_buffer.pkl
_train_stage4_YYYYMMDD-HHMMSS.txt
_figure.png
_model_configuration_YYYYMMDD-HHMMSS.txt
```

重点看：

```text
_train_stage4_*.txt: 每个 episode 的 reward、结果、步数、loss
_figure.png: 训练曲线
*.pt: PyTorch 权重
stage*_latest_buffer.pkl: replay buffer
```

## 17. 快速查看训练日志

列出模型目录：

```bash
find src/turtlebot3_drl/model -maxdepth 3 -type d | sort
```

找训练日志：

```bash
find src/turtlebot3_drl/model -name '_train_stage*.txt'
```

查看最后 20 行：

```bash
tail -n 20 src/turtlebot3_drl/model/<HOSTNAME>/<MODEL_NAME>/_train_stage4_*.txt
```

日志列含义：

```text
episode
reward
success/outcome
duration
steps
total_steps
memory length
avg_critic_loss
avg_actor_loss
```

outcome 编号：

```text
0 UNKNOWN
1 SUCCESS
2 COLLISION_WALL
3 COLLISION_OBSTACLE
4 TIMEOUT
5 TUMBLE
```

## 18. 用 Python 画 reward 曲线

在宿主机或容器里执行：

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

logs = list(Path("src/turtlebot3_drl/model").glob("**/_train_stage*.txt"))
if not logs:
    raise SystemExit("No training logs found")

log = sorted(logs)[-1]
print("using log:", log)

df = pd.read_csv(log, skipinitialspace=True)
df["reward_ma20"] = df["reward"].rolling(20, min_periods=1).mean()

plt.figure(figsize=(12, 5))
plt.plot(df["episode"], df["reward"], alpha=0.25, label="reward")
plt.plot(df["episode"], df["reward_ma20"], label="reward moving average 20")
plt.xlabel("episode")
plt.ylabel("reward")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("reward_curve.png", dpi=160)
print("saved reward_curve.png")
PY
```

下载或打开：

```bash
ls -lh reward_curve.png
```

## 19. 打包训练成果

训练会产生较大文件，尤其是 replay buffer。建议每次重要训练完成后打包：

```bash
tar -czf drlnav_results_$(date +%Y%m%d_%H%M%S).tar.gz src/turtlebot3_drl/model
```

如果只想保存模型权重和日志，不保存 replay buffer：

```bash
tar --exclude='*latest_buffer.pkl' \
  -czf drlnav_results_light_$(date +%Y%m%d_%H%M%S).tar.gz \
  src/turtlebot3_drl/model
```

## 20. 推荐第一次完整流程

第一次不要直接开 4 个并行环境。建议按这个顺序：

```text
1. nvidia-smi 通过
2. docker run --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi 通过
3. docker build -t turtlebot3_drlnav . 成功
4. 单环境 stage4 + td3 跑通
5. 确认 model 目录出现 _train_stage4_*.txt 和 .pt 权重
6. 测试恢复训练命令
7. 再启动 parallel_train_agent + 2 个 sampler worker
8. 稳定后再加到 3 或 4 个 worker
```

## 21. 常见问题

### docker: command not found

Docker 没装好，回到第 3 节。

### docker: permission denied

当前用户不在 docker 组：

```bash
sudo usermod -aG docker $USER
```

然后退出 SSH，重新登录。

### 容器里 torch.cuda.is_available() 是 False

先在宿主机验证：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi
```

如果第二个失败，检查 NVIDIA Container Toolkit。

### Gazebo 窗口打不开

检查：

```bash
echo $DISPLAY
xhost +local:docker
```

如果是云服务器，需要先配置 VNC、NoMachine、NICE DCV 或 X11 forwarding。

### Make sure to launch the gazebo simulation node first

先启动：

```bash
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py
```

再启动 `gazebo_goals`、`environment`、`train_agent`。

### env step service not available

`environment` 没启动，或者 `ROS_DOMAIN_ID` 不一致。

检查：

```bash
echo $ROS_DOMAIN_ID
ros2 service list | grep step_comm
```

### 并行 worker 互相串台

每个 worker 必须使用不同的：

```text
ROS_DOMAIN_ID
GAZEBO_MASTER_URI
container name
hostname
```

### 训练很慢

优先检查：

```bash
nvidia-smi
top
docker stats
```

Gazebo 通常吃 CPU，神经网络训练吃 GPU。并行 worker 太多时，CPU 可能先满。

### 训练结果没有变好

先确认：

```text
是否还在 OBSERVE_STEPS 随机探索阶段
reward 是否有明显震荡或逐步提升
SUCCESS 比例是否增加
collision 和 timeout 是否下降
```

强化学习训练可能需要很多 episode。不要只看前几十个 episode 下结论。

## 22. 外部参考

- Docker Engine Ubuntu 官方安装文档：https://docs.docker.com/engine/install/ubuntu/
- NVIDIA Container Toolkit 官方安装文档：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- 本项目 GitHub 分支：https://github.com/Wither033/turtlebot3_drlnav/tree/codex/parallel-sampler-training
- 原项目上游仓库：https://github.com/tomasvr/turtlebot3_drlnav
