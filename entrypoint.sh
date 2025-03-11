#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/conda/envs/env_py311_p5
exec "$@"