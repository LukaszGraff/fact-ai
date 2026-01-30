FROM python:3.10.14-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC

# Install apt packages.
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends \
        libosmesa6-dev \
        libgl1-mesa-glx \
        libglew-dev \
        libglfw3 \
        man-db \
        vim \
        wget \
        tmux \
        git \
        patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# Install MuJoCo 2.1.0 for D4RL.
ARG MUJOCO_BASE_PATH=/root/.mujoco
ENV MUJOCO_PY_MUJOCO_PATH=$MUJOCO_BASE_PATH/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PY_MUJOCO_PATH/bin
RUN mkdir -p $MUJOCO_PY_MUJOCO_PATH \
    && wget -q -P /tmp https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
    && tar --no-same-owner -xf /tmp/mujoco210-linux-x86_64.tar.gz -C $MUJOCO_BASE_PATH

RUN pip install -U pip \
    && pip install nvitop
