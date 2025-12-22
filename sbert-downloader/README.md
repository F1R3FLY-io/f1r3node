# SBERT-downloader

This sets up a helper script to download an SBERT model (see [`downloader.py`](./downloader.py)) and save it locally. The script is meant to be run with the provided [Dockerfile](./Dockerfile), which should be composed as a service (see: [`shard-with-autopropose.yml`](../docker/shard-with-autopropose.yml)). In this way, docker-compose would be able to run the script in a container to download the image and share it via a volume.

See the docker-compose file linked above to see where the model is stored. This path will be needed to set the ENV var `SBERT_PATH` for use by the Rholang interpreter.

NOTE: The downloaded pytorch is set to the CPU only index-url to avoid downloading the CUDA library alongside (which takes up a lot of space).
