FROM debian:bookworm-slim AS nfqws-build
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gcc make libcap-dev libnetfilter-queue-dev \
    libnfnetlink-dev libmnl-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://codeload.github.com/bol-van/zapret/tar.gz/refs/tags/v72.13 -o /tmp/zapret.tar.gz \
    && echo '04696a1b6fe766e35cd6426555520c79d5052a759cdf4d7724a1f46eba4670f6  /tmp/zapret.tar.gz' | sha256sum -c - \
    && mkdir /src && tar -xzf /tmp/zapret.tar.gz --strip-components=1 -C /src \
    && make -C /src/nfq

FROM python:3.11-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates ffmpeg iptables libcap2 libmnl0 libnetfilter-queue1 \
    libnfnetlink0 libopus0 mawk util-linux zlib1g \
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
RUN python -m pip install --no-cache-dir -r /app/requirements.txt
COPY bot.py startBot.py /app/
COPY modules/ /app/modules/
COPY utils/ /app/utils/
COPY voice_interface/ /app/voice_interface/
CMD ["python", "/app/startBot.py", "false"]
