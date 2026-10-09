"""AITW Monomer Surrogate CLI commands"""
import os
import sys

from .main import cli, click


@cli.command()
@click.option(
    '--data_file', type=click.Path(exists=True, dir_okay=False), default="surrogate_data.csv",
    help='Path to the CSV file containing the training data'
)
@click.option('--n_epochs', type=int, default=10000, help='Number of epochs to train the model')
@click.option(
    '--patience', type=int, default=500,
    help='Number of epochs to wait before stopping the training if no improvement is observed'
)
# @click.option(
#     '--loss_curve_file', type=click.Path(), default="loss_curve.dat",
#     help='Path to the file where the loss curve will be saved'
# )
# @click.option(
#     '--test_losses_file', type=click.Path(), default="test_losses.dat",
#     help='Path to the file where the test losses will be saved'
# )
# @click.option(
#     '--test_file', type=click.Path(), default="test.dat",
#     help='Path to the file where the test results will be saved'
# )
# @click.option(
#     '--output_weights_file', type=click.Path(), default="weights.pt",
#     help='Path to the file where the trained weights will be saved'
# )
@click.option(
    '--train_test_split_ratio', type=float, default=0.2,
    help='Ratio of the training data to the test data'
)
@click.option(
    '--val_test_split_ratio', type=float, default=0.5,
    help='Ratio of the validation data to the test data'
)
@click.option(
    '--train_batch_size', type=int, default=100,
    help='Batch size for training'
)
@click.option(
    '--output_dir', type=click.Path(), default="output",
    help='Output directory'
)
def train(
        data_file: str = "surrogate_data.csv",
        n_epochs: int = 10000,
        patience: int = 500,
        # loss_curve_file: str = "loss_curve.dat",
        # test_losses_file: str = "test_losses.dat",
        # test_file: str = "test.dat",
        # output_weights_file: str = "weights.pt",
        train_test_split_ratio: float = 0.2,
        val_test_split_ratio: float = 0.5,
        train_batch_size: int = 100,
        output_dir: str = "output"
    ) -> None:
    """Train wood microstructure model"""
    from aitw_monomer_surrogate.train_surrogate import train_surrogate

    # training_params = {}
    # if config_file:
    #     with open(config_file, 'r') as f:
    #         training_params = json.load(f)

    if os.path.exists(output_dir):
        click.echo(f"Output directory '{output_dir}' already exists. Please choose a different directory.", err=True)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    loss_curve_file = os.path.join(output_dir, "loss_curve.dat")
    test_losses_file = os.path.join(output_dir, "test_losses.dat")
    test_file = os.path.join(output_dir, "test.dat")
    output_weights_file = os.path.join(output_dir, "weights.pt")

    train_surrogate(
        n_epochs=n_epochs,
        patience=patience,
        data_file=data_file,
        loss_curve_file=loss_curve_file,
        test_losses_file=test_losses_file,
        test_file=test_file,
        output_weights_file=output_weights_file,
        train_test_split_ratio=train_test_split_ratio,
        val_test_split_ratio=val_test_split_ratio,
        train_batch_size=train_batch_size,
    )


__all__ = [
    'train'
]
