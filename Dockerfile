FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY . .

RUN chmod -R a+w results/

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

ENV HOME="/tmp"
ENV USER="mvaa"

RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab

# Left for testing purposes
CMD ["python", "mvaa/tests/reproduce_all.py"]
