FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    dosbox-x \
    xvfb \
    xdotool \
    x11-apps \
    x11vnc \
    libxcb-render0 \
    libxcb-shm0 \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=8
RUN pip install --no-cache-dir mss numpy pillow gymnasium matplotlib
# refactor (2026-09-27): in-process XTEST key input (xdotool spawns a process
# per key: 23-32 ms) and fast connected-component labeling
RUN pip install --no-cache-dir python-xlib scipy
# Plain `pip install torch` on linux_aarch64 grabs the latest release,
# which hard-requires CUDA shared libs (libcudart.so etc.) even to
# import -- multi-GB nvidia-* packages, useless and broken on this
# GPU-less container (confirmed: torch 2.14 fails `import torch`
# without them). torch==2.3.1 predates that and is genuinely CPU-only.
# --no-deps on both torch and sb3 so pip never "fixes" a version
# mismatch by silently pulling nvidia-* back in; sb3's warning about
# wanting torch>=2.8 is harmless here (verified: PPO/CnnPolicy trains
# fine on 2.3.1).
RUN pip install --no-cache-dir --no-deps 'torch==2.3.1'
RUN pip install --no-cache-dir filelock typing-extensions sympy networkx jinja2 fsspec
RUN pip install --no-cache-dir --no-deps stable-baselines3
RUN pip install --no-cache-dir cloudpickle pandas

WORKDIR /work
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV DISPLAY=:99
EXPOSE 5900
ENTRYPOINT ["/entrypoint.sh"]
