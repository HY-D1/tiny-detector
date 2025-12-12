FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project metadata + source (kept small by .dockerignore)
COPY pyproject.toml setup.cfg /app/
COPY src /app/src
COPY app /app/app
COPY tests /app/tests

# Install editable so `import src` works everywhere
RUN pip install -e .

CMD ["/bin/bash"]
