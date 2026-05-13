ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DRIVER_VERSION="854ff04"

RUN groupadd -f render && usermod -aG render root

WORKDIR /ryzers
ENV DEBIAN_FRONTEND=noninteractive

# Copy build scripts
COPY build_xdna.sh /ryzers/build_xdna.sh
COPY extract_npu_firmware.sh /ryzers/extract_npu_firmware.sh
RUN chmod +x /ryzers/build_xdna.sh /ryzers/extract_npu_firmware.sh

# Build XRT + XDNA with dynamic dep resolution
RUN /ryzers/build_xdna.sh $DRIVER_VERSION

# Extract NPU firmware from Ubuntu's linux-firmware deb
RUN /ryzers/extract_npu_firmware.sh

ENV SHELL=/bin/bash

COPY test.sh /ryzers/
CMD ["/ryzers/test.sh"]
