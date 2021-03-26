FROM ubuntu:latest
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl gnupg2 ca-certificates && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        edgetpu-compiler
