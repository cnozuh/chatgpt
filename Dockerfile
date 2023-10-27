FROM python:3.8-slim-bullseye

WORKDIR /app

RUN rm -f /etc/apt/sources.list.d/*.list

RUN apt-get update \
    && apt-get install -y \
        sudo \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
    && chmod -R 777 $HOME

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app