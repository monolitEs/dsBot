FROM debian:bookworm-slim AS nfqws-build
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gcc make libcap-dev libnetfilter-queue-dev \
    libnfnetlink-dev libmnl-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://codeload.github.com/bol-van/zapret/tar.gz/refs/tags/v72.13 -o /tmp/zapret.tar.gz \
    && echo '04696a1b6fe766e35cd6426555520c79d5052a759cdf4d7724a1f46eba4670f6  /tmp/zapret.tar.gz' | sha256sum -c - \
    && mkdir /src && tar -xzf /tmp/zapret.tar.gz --strip-components=1 -C /src \
    && make -C /src/nfq

FROM debian:bookworm-slim AS stt-models
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /models/gigaam-ctc
ENV GIGAAM_SOURCE=https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-punct-giga-am-v3-russian-2025-12-16/resolve/4fb5407ff028a69fec516cdf4c10fac9ddea7c16
RUN curl -fsSL "$GIGAAM_SOURCE/model.int8.onnx" -o model.int8.onnx \
    && echo 'd5fea8df94263c285e54b21e5774b707c707192d3bdbeffd7b1eb07fb6743b35  model.int8.onnx' | sha256sum -c -
RUN curl -fsSL "$GIGAAM_SOURCE/tokens.txt" -o tokens.txt \
    && echo '142de7570b3de5b3035ce111a89c228e80e6085273731d944093ddf24fa539cd  tokens.txt' | sha256sum -c -
RUN curl -fsSL "$GIGAAM_SOURCE/LICENSE" -o LICENSE \
    && echo 'f00de6715714c7a63d08639cdbfaa40224eefc407302614bd19f1a8b98c875aa  LICENSE' | sha256sum -c -

FROM python:3.11-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates ffmpeg iptables libcap2 libmnl0 libnetfilter-queue1 \
    libnfnetlink0 libopus0 libgomp1 mawk util-linux zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 bot && useradd --uid 10001 --gid bot --home-dir /data bot \
    && install -d -o bot -g bot /data
COPY --from=nfqws-build /src/nfq/nfqws /usr/local/bin/nfqws
COPY --from=nfqws-build /src/docs/LICENSE.txt /usr/share/doc/nfqws/LICENSE.txt
COPY docker/vendor/zapret-alt13/ /opt/zapret/
COPY docker/nfqws.conf /etc/zapret/nfqws.conf
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/bot-entrypoint
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/data
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/bot-entrypoint"]

FROM runtime AS bot
COPY requirements.txt /app/requirements.txt
COPY docker/vendor/pycord/ /app/docker/vendor/pycord/
RUN cd /app && python -m pip install --no-cache-dir -r requirements.txt
COPY --from=stt-models /models/ /opt/models/
ENV GIGAAM_MODEL_DIR=/opt/models/gigaam-ctc
COPY models/silero_vad.onnx /app/models/silero_vad.onnx
COPY bot.py startBot.py /app/
COPY modules/ /app/modules/
COPY utils/ /app/utils/
COPY voice_interface/ /app/voice_interface/
CMD ["python", "/app/startBot.py", "false"]
