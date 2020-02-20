FROM continuumio/miniconda3

# Activate shell and install linux deps
SHELL ["/bin/bash", "-c"]
ENV PATH="/opt/conda/bin/:$PATH"
RUN apt-get update && \
    apt install -y gcc && \
    mkdir -p /home/beep_ep && \
    conda create -n beep python=3.6

WORKDIR /home/beep_ep

# Create BEEP_EP env
ENV PATH="/opt/conda/envs/beep/bin:$PATH"
ENV TQDM_OFF=1

COPY . /home/beep_ep/

# Add version tag from ENV or build args
ARG BEEP_EP_VERSION_TAG="docker-built"
ENV BEEP_EP_VERSION_TAG=$BEEP_EP_VERSION_TAG

# Install beep_ep
RUN source /opt/conda/bin/activate beep && \
    pip install -e .[tests] && \
    chmod +x dockertest.sh
