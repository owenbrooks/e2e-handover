ARG from
FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu20.04 as nvidia
FROM ${from}

COPY --from=nvidia /usr/local /usr/local
COPY --from=nvidia /etc/ld.so.conf.d /etc/ld.so.conf.d

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
# bulid ith --build-arg from=ghcr.io/owenbrooks/e2e-handover:main