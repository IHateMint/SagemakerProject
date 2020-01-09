FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN apt-get update \
    && apt-get -y install --no-install-recommends git openssh-client gcc curl nginx-light vim \
    && mkdir -p -m 0600 ~/.ssh \
    && mkdir -p -m 0600 ~/.aws \
    && ssh-keyscan -t rsa github.com > ~/.ssh/known_hosts \
    && pip install --upgrade pip \
    && pip install pandas \
    && pip install boto3 \
    && pip install sagemaker

ENV PATH="/opt/ml/code:${PATH}"

# Copies the training code inside the container
COPY /ponet /opt/ml/code
WORKDIR /opt/ml/code