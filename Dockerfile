FROM --platform=linux/amd64 pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime AS hecktor2026-task

ENV PYTHONUNBUFFERED=1
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

ENV PATH=/home/user/.local/bin:$PATH

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --no-color --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/
ENTRYPOINT ["python", "inference.py"]
