FROM continuumio/miniconda3

# 必要なパッケージのインストール（SHELL を変更する前に実行）
RUN apt-get update && apt-get install -y \
    make \
    sudo \
    wget \
    vim \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Conda 環境パスを正しく設定
ENV PATH="/opt/miniconda3/bin:$PATH"

# Conda仮想環境の作成（environment.ymlを使う）
COPY environment.yml /tmp/environment.yml

WORKDIR /opt
# 環境を /opt/conda/envs に明示的に作成
RUN conda env create -p /opt/conda/envs/env_py311_p5 -f /tmp/environment.yml

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# MLflow サーバーを起動
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////works/mlruns/mlflow.db", "--default-artifact-root", "/works/mlruns", "--host", "0.0.0.0", "--port", "5000"]

# ホストの works/ ディレクトリをコンテナの /works にコピーする
COPY ./works /works

# 作業ディレクトリの設定
WORKDIR /works