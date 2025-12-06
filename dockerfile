FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

WORKDIR /chatbot

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1 
ENV POETRY_HOME="/opt/poetry" 
ENV PATH="${POETRY_HOME}/bin:${PATH}"
ENV POETRY_VERSION=2.2.1 
ENV POETRY_NO_INTERACTION=1 
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_CACHE_DIR="/var/cache/pypoetry"

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip supervisor \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && ln -sf /usr/bin/python3 /bin/python \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && pip install "poetry==$POETRY_VERSION" \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

COPY . .

RUN poetry install --only main --no-cache

# Clean up cache to reduce image size
RUN rm -rf $POETRY_HOME/lib $POETRY_CACHE_DIR /root/.cache/pip /tmp/*

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# default FastAPI server port
EXPOSE 8000     

# default Streamlit UI port 
EXPOSE 8500


CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]