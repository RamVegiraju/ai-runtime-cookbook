# The AWS AIR base supplies Python, uv, CUDA, and the platform networking libraries.
FROM databricksruntime/air:dcs-base-aws-runtime

# River is absent from the base image. Install it in the base Python environment.
RUN uv pip install --no-cache river==0.22.0

# AIR does not honor WORKDIR; workload.yaml runs this absolute path.
COPY train.py /app/train.py
