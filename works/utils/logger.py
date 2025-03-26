import mlflow

# 基本情報
def setup_mlflow(cfg):
    # MLflow の実験を設定
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    # MLflow のトラッキング URI を設定
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # タグを追加
    mlflow.set_tag("model_name", cfg.train.model_name)
    mlflow.set_tag("dataset", cfg.dataset.dataset_name)
    mlflow.set_tag("optimizer", cfg.train.optimizer)
    mlflow.set_tag("experiment", cfg.mlflow.experiment)

# ハイパラメータのログ
def parameter_info(cfg):
    # ハイパラメータの記録
    mlflow.log_param("learning_rate", cfg.train.lr)
    mlflow.log_param("batch_size", cfg.train.batch_size)
    mlflow.log_param("num_epochs", cfg.train.epochs)
    mlflow.log_param("loss_function", cfg.train.criterion)
    mlflow.log_param("optimizer", cfg.train.optimizer)

def log_metrics(epoch, train_loss, train_acc, val_loss, val_acc):
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_acc", train_acc, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_acc", val_acc, step=epoch)

def best_log(best_val_loss, best_acc):
    # 最終結果だけもう一度ログ
    mlflow.log_metric("best_val_loss", best_val_loss)
    mlflow.log_metric("best_val_acc", best_acc)

# データセットに関するログ
def log_dataset_info(cfg, train_loader, val_loader):
    # データセットの情報を記録
    mlflow.log_param("dataset_path", cfg.dataset.metadata_csv)
    mlflow.log_param("num_train_samples", len(train_loader.dataset))
    mlflow.log_param("num_val_samples", len(val_loader.dataset))