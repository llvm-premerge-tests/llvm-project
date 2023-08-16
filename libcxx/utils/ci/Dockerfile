#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

#
# This Dockerfile describes the base image used to run the various libc++
# build bots. By default, the image runs the Buildkite Agent, however one
# can also just start the image with a shell to debug CI failures.
#
# To start a Buildkite Agent, run it as:
#   $ docker run --env-file <secrets> -it $(docker build -q libcxx/utils/ci)
#
# The environment variables in `<secrets>` should be the ones necessary
# to run a BuildKite agent:
#
#   BUILDKITE_AGENT_TOKEN=<token>
#
# If you're only looking to run the Docker image locally for debugging a
# build bot, see the `run-buildbot-container` script located in this directory.
#
# A pre-built version of this image is maintained on DockerHub as ldionne/libcxx-builder.
# To update the image, rebuild it and push it to ldionne/libcxx-builder (which
# will obviously only work if you have permission to do so).
#
#   $ docker build -t ldionne/libcxx-builder libcxx/utils/ci
#   $ docker push ldionne/libcxx-builder
#

FROM ubuntu:jammy

# Make sure apt-get doesn't try to prompt for stuff like our time zone, etc.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y bash curl

# Install various tools used by the build or the test suite
#RUN apt-get update && apt-get install -y ninja-build python3 python3-sphinx python3-distutils python3-psutil git gdb ccache
# TODO add ninja-build once 1.11 is available in Ubuntu, also remove the manual installation.
RUN apt-get update && apt-get install -y python3 python3-sphinx python3-distutils python3-psutil git gdb ccache
RUN apt-get update && apt-get install -y wget && \
  wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip && \
  gunzip /usr/local/bin/ninja.gz && \
  chmod a+x /usr/local/bin/ninja

# Install dependencies required to run the LLDB data formatter tests
RUN apt-get update && apt-get install -y python3 python3-dev libpython3-dev uuid-dev libncurses5-dev swig3.0 libxml2-dev libedit-dev

# Locales for gdb and localization tests
RUN apt-get update && apt-get install -y language-pack-en language-pack-fr \
                                         language-pack-ja language-pack-ru \
                                         language-pack-zh-hans
# These two are not enabled by default so generate them
RUN printf "fr_CA ISO-8859-1\ncs_CZ ISO-8859-2" >> /etc/locale.gen
RUN mkdir /usr/local/share/i1en/
RUN printf "fr_CA ISO-8859-1\ncs_CZ ISO-8859-2" >> /usr/local/share/i1en/SUPPORTED
RUN locale-gen

# Install Clang <latest>, <latest-1> and ToT, which are the ones we support.
# We also install <latest-2> because we need to support the "latest-1" of the
# current LLVM release branch, which is effectively the <latest-2> of the
# tip-of-trunk LLVM. For example, after branching LLVM 14 but before branching
# LLVM 15, we still need to have Clang 12 in this Docker image because the LLVM
# 14 release branch CI uses it. The tip-of-trunk CI will never use Clang 12,
# though.
# LLVM POST-BRANCH bump version
ENV LLVM_HEAD_VERSION=17
RUN apt-get update && apt-get install -y lsb-release wget software-properties-common
RUN wget https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh
RUN bash /tmp/llvm.sh $(($LLVM_HEAD_VERSION - 3)) # for CI transitions
RUN bash /tmp/llvm.sh $(($LLVM_HEAD_VERSION - 2)) # previous release
RUN bash /tmp/llvm.sh $(($LLVM_HEAD_VERSION - 1)) # latest release
RUN bash /tmp/llvm.sh $LLVM_HEAD_VERSION          # current ToT

# Install clang-scan-deps, which is required to build modules; always all supported versions.
RUN apt-get update && apt-get install -y clang-tools-$(($LLVM_HEAD_VERSION - 3)) \
                                         clang-tools-$(($LLVM_HEAD_VERSION - 2)) \
                                         clang-tools-$(($LLVM_HEAD_VERSION - 1)) \
                                         clang-tools-$LLVM_HEAD_VERSION

# Install clang-format; always use the lastest stable branch.
RUN apt-get update && apt-get install -y clang-format-$(($LLVM_HEAD_VERSION - 2)) clang-format-$(($LLVM_HEAD_VERSION - 1))

# Install clang-tidy
# TODO(LLVM-17) revert D148831 to only install $(($LLVM_HEAD_VERSION - 1)) and $LLVM_HEAD_VERSION
# The usage of the ToT version is needed due to module issues with Clang 16
RUN apt-get update && apt-get install -y clang-tidy-$(($LLVM_HEAD_VERSION - 2)) clang-tidy-$(($LLVM_HEAD_VERSION - 1)) clang-tidy-$LLVM_HEAD_VERSION

# Install llvm-dev and libclang-dev to compile custom clang-tidy checks
# TODO(LLVM-17) revert D148831 to only install $(($LLVM_HEAD_VERSION - 1)) and $LLVM_HEAD_VERSION
# The usage of the ToT version is needed due to module issues with Clang 16
RUN apt-get update && apt-get install -y llvm-$(($LLVM_HEAD_VERSION - 2))-dev llvm-$(($LLVM_HEAD_VERSION - 1))-dev     llvm-$LLVM_HEAD_VERSION-dev \
                                     libclang-$(($LLVM_HEAD_VERSION - 2))-dev libclang-$(($LLVM_HEAD_VERSION - 1))-dev libclang-$LLVM_HEAD_VERSION-dev \
                                                                                                                       libomp5-$LLVM_HEAD_VERSION

# Install the most recent GCC, like clang install the previous version as a transition.
ENV GCC_LATEST_VERSION=13
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt install -y gcc-$((GCC_LATEST_VERSION - 1)) g++-$((GCC_LATEST_VERSION - 1))
RUN apt-get update && apt install -y gcc-$GCC_LATEST_VERSION g++-$GCC_LATEST_VERSION

# Remove all, no longer needed, apt data
RUN rm -rf /var/lib/apt/lists/*

# Install a recent CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-linux-x86_64.sh -O /tmp/install-cmake.sh
RUN bash /tmp/install-cmake.sh --prefix=/usr --exclude-subdir --skip-license
RUN rm /tmp/install-cmake.sh

# Install a newer CMake for modules
# TODO Remove the duplicated installation when all runtimes can be build with CMake 3.27.
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.0-rc2/cmake-3.27.0-rc2-linux-x86_64.sh -O /tmp/install-cmake.sh
RUN bash /tmp/install-cmake.sh --prefix=/opt --exclude-subdir --skip-license
RUN rm /tmp/install-cmake.sh

# Install the Android platform tools (e.g. adb) into /opt/android/sdk.
RUN apt-get update && apt-get install -y unzip
RUN mkdir -p /opt/android/sdk && cd /opt/android/sdk && \
    curl -LO https://dl.google.com/android/repository/platform-tools-latest-linux.zip && \
    unzip platform-tools-latest-linux.zip && \
    rm platform-tools-latest-linux.zip

# Install the current Android compiler.
ENV ANDROID_CLANG_VERSION=r498229
RUN git clone --filter=blob:none --sparse \
        https://android.googlesource.com/platform/prebuilts/clang/host/linux-x86 \
        /opt/android/clang && \
    git -C /opt/android/clang sparse-checkout add clang-${ANDROID_CLANG_VERSION} && \
    rm -fr /opt/android/clang/.git && \
    ln -sf /opt/android/clang/clang-${ANDROID_CLANG_VERSION} /opt/android/clang/clang-current

# Install an Android sysroot. New AOSP sysroots are available at
# https://ci.android.com/builds/branches/aosp-master/grid, the "ndk" target. The
# NDK also makes its sysroot prebuilt available at
# https://android.googlesource.com/platform/prebuilts/ndk/+/refs/heads/dev/platform/sysroot.
ENV ANDROID_SYSROOT_BID=10375398
RUN cd /opt/android && \
    curl -L -o ndk_platform.tar.bz2 \
        https://androidbuildinternal.googleapis.com/android/internal/build/v3/builds/${ANDROID_SYSROOT_BID}/ndk/attempts/latest/artifacts/ndk_platform.tar.bz2/url && \
    tar xf ndk_platform.tar.bz2 && \
    rm ndk_platform.tar.bz2

# Install Docker. Mark the binary setuid so it can be run without prefixing it
# with sudo. Adding the container user to the docker group doesn't work because
# /var/run/docker.sock is owned by the host's docker GID, not the container's
# docker GID.
RUN apt-get update && apt-get install -y gpg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
         "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
         tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
    chmod u+s /usr/bin/docker

COPY ./container-setup.sh /opt/bin/container-setup.sh

# Change the user to a non-root user, since some of the libc++ tests
# (e.g. filesystem) require running as non-root. Also setup passwordless sudo.
RUN apt-get update && apt-get install -y sudo
RUN echo "ALL ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN useradd --create-home libcxx-builder
USER libcxx-builder
WORKDIR /home/libcxx-builder

# Add Android platform-tools to the PATH.
ENV PATH="/opt/android/sdk/platform-tools:${PATH}"

# Install the Buildkite agent and dependencies. This must be done as non-root
# for the Buildkite agent to be installed in a path where we can find it.
RUN bash -c "$(curl -sL https://raw.githubusercontent.com/buildkite/agent/main/install.sh)"
ENV PATH="${PATH}:/home/libcxx-builder/.buildkite-agent/bin"

# By default, start the Buildkite agent (this requires a token).
CMD /opt/bin/container-setup.sh && buildkite-agent start
