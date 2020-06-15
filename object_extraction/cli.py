"""object_extraction.cli."""


import click
import sys
sys.path.append("..")
from object_extraction.encoding.predict_model_pv import predict_process_pv
from object_extraction.encoding.predict_model import predict_process
from object_extraction.encoding.train_model import train_process
from object_extraction.encoding.train_model_mosaic import train_process_mosaic

@click.group(short_help="COGEO predict")
@click.version_option(version="1.0.0", message="%(version)s")
def object_extraction():
    """Rasterio predict subcommands."""
    pass
@object_extraction.command(short_help="Use AI Model to Extract Object.")
@click.option(
    "--model_path",
    multiple=False,
    type=str,
    help="obs model path",
)
@click.option(
    "--img_path",
    multiple=False,
    type=str,
    help="image's obs path for predict",
)
@click.option(
    "--img_obs_path",
    multiple=False,
    type=str,
    help="image's obs path for predict",
)
@click.option(
    "--output_path",
    "--o",
    multiple=False,
    type=str,
    help="the output path for result",
)
def predict(model_path, img_obs_path, img_path, output_path):
    """Use AI Model to Extract Object."""
    predict_process_pv(model_path=model_path, img_path=img_path, img_obs_path=img_obs_path, output_path=output_path)


@object_extraction.command(short_help="Use AI Model to Extract Object.")
@click.option(
    "--model_path",
    multiple=False,
    type=str,
    help="obs model path",
)
@click.option(
    "--cat_ids",
    multiple=False,
    type=str,
    help="image's catalog ids for predict",
)
@click.option(
    "--paths",
    multiple=False,
    type=str,
    help="image's obs path for predict",
)
@click.option(
    "--output_path",
    "--o",
    multiple=False,
    type=str,
    help="the output path for result",
)
def predict_mosaic(output_path, model_path, cat_ids=None, paths=None):
    """Use AI Model to Extract Object."""
    predict_process(model_path=model_path, output_path=output_path, cat_ids=cat_ids, paths=paths)


@object_extraction.command(short_help="Train a Model for extracting Objects.")
@click.option(
    "--data_folder",
    multiple=False,
    type=str,
    help="image's obs path for train",
)
@click.option(
    "--pre_train_model",
    multiple=False,
    type=str,
    help="pre_train_model's obs path for train",
)
@click.option(
    "--epochs",
    multiple=False,
    type=str,
    help="train epochs",
)
@click.option(
    "--output_path",
    "--o",
    multiple=False,
    type=str,
    help="the output path for result",
)
def train(data_folder, output_path, epochs, pre_train_model):
    """Use AI Model to Extract Object."""
    train_process(data_folder=data_folder,output_path=output_path, epochs=epochs, pre_train_model=pre_train_model)

@object_extraction.command(short_help="Train a Model for extracting Objects.")
@click.option(
    "--img_paths",
    multiple=False,
    type=str,
    help="image's obs path for train",
)
@click.option(
    "--label_paths",
    multiple=False,
    type=str,
    help="label's obs path for train",
)
@click.option(
    "--div_ratio",
    multiple=False,
    type=str,
    help="train/val dataset div ratio",
)
@click.option(
    "--pre_train_model",
    multiple=False,
    type=str,
    help="pre_train_model's obs path for train",
)
@click.option(
    "--epochs",
    multiple=False,
    type=str,
    help="train epochs",
)
@click.option(
    "--output_path",
    "--o",
    multiple=False,
    type=str,
    help="the output path for result",
)
def train_mosaic(img_paths, label_paths, div_ratio, output_path, epochs, pre_train_model):
    """Use AI Model to Extract Object."""
    train_process_mosaic(img_paths=img_paths, label_paths=label_paths, div_ratio=div_ratio, output_path=output_path,
                         epochs=epochs, pre_train_model=pre_train_model)