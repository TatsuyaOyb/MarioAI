ARG BASE_IMAGE=nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG USERNAME=TatsuyaOyb
# TODO: User should be refactored instead of hard coded olympus

USER root

ENV DEBIAN_FRONTEND noninteractive

ENV USERNAME ${USERNAME}
ENV GROUPNAME ${USERNAME}
ENV UID 1000
ENV GID 1000
ENV HOME /home/${USERNAME}
ENV PATH ${HOME}/.local/bin:${PATH}
ENV POETRY_DIR ${HOME}/poetry-mario
# TODO: UID and GID be refactored instead of hard coded 1000 to your host UID

# Use bash instead of sh
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
	apt-get install -yq --no-install-recommends \
		apt-transport-https \
		build-essential \
		bzip2 \
		ca-certificates \
		curl \
		g++ \
		git \
		gnupg \
		graphviz \
		locales \
		lsb-release \
		openssh-client \
		sudo \
		unzip \
		vim \
		wget \
		zip \
		python3.10 \
		python3-pip \
		python3-dev \
		python3-setuptools \
        # for GUI
		libssl-dev zlib1g-dev libbz2-dev \
		libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev \
		# for gym env.render()
		#libglib2.0-dev libsm6 libxext6 libxrender-dev \
		freeglut3-dev ffmpeg \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# change the locale from POSIX to UTF-8(Japanese)
RUN locale-gen en_US en_US.UTF-8 && \
	update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LANGUAGE=en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

# Create USERNAME user with UID=1000 and in the 'users' group
# but allow for non-initial launches of the notebook to have
# ${HOME} provided by the contents of a PV
RUN useradd -M -s /bin/bash -N -u $UID $USERNAME && \
    groupadd -g ${GID} ${GROUPNAME} && \
    usermod -g ${GROUPNAME} ${USERNAME} && \
    chown -R ${USERNAME}:${GROUPNAME} /usr/local/bin && \
    mkdir -p ${HOME}/workspace && \
    chown -R ${USERNAME}:${GROUPNAME} ${HOME}


# NOTE: Beyond this point be careful of breaking out
# or otherwise adding new layers with RUN, chown, etc.
# The image size can grow significantly.

# python to 3.10
RUN apt-get update && apt-get install -yq --no-install-recommends \
        python-is-python3 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies for glvnd and X11
RUN apt-get update && apt-get install -yq --no-install-recommends \
        libglvnd0 libgl1 libglx0 libegl1 \
        libglew2.2 libgl1-mesa-glx libosmesa6 \
		x11-apps \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install base python3 packages
RUN curl -sSL https://install.python-poetry.org | \
    POETRY_HOME=/etc/poetry POETRY_VERSION=1.2.0 python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    mkdir -p ${POETRY_DIR}
COPY ./docker/pyproject.toml ${POETRY_DIR}/pyproject.toml
COPY ./docker/scripts/poetry_install.sh ${POETRY_DIR}/poetry_install.sh
WORKDIR /opt/app
RUN cd ${POETRY_DIR} && bash poetry_install.sh

# Time Zone
ENV TIMEZONE Asia/Toyko
RUN ln -snf /usr/share/zoneinfo/$TIMEZONE /etc/localtime && \
    echo $TIMEZONE > /etc/timezone

# Configure container startup
RUN chown -R ${USERNAME}:${GROUPNAME} /usr/local/bin && \
	chown -R ${USERNAME}:${GROUPNAME} ${HOME} && \
    chmod -R 755 ${HOME}
EXPOSE 8888
USER ${USERNAME}
WORKDIR ${HOME}/workspace
CMD ["bash"]