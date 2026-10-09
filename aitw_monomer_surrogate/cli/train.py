"""AITW Monomer Surrogate CLI commands"""
import logging

from .main import cli, click

verbose_map = {
    0: logging.INFO,
    1: logging.DEBUG,
}


@cli.command()
@click.option(
    '--data_file', type=click.Path(exists=True, dir_okay=False), required=True, show_default=True,
    help='Path to the CSV file containing the training data'
)
@click.option(
    '--output_dir', type=click.Path(), default="output", show_default=True,
    help='Output directory'
)
@click.option('--n_epochs', type=int, default=10000,  show_default=True, help='Number of epochs to train the model')
@click.option(
    '--patience', type=int, default=500, show_default=True,
    help='Number of epochs to wait before stopping the training if no improvement is observed'
)
@click.option(
    '--train_test_split_ratio', type=float, default=0.2, show_default=True,
    help='Ratio of the training data to the test data'
)
@click.option(
    '--val_test_split_ratio', type=float, default=0.5, show_default=True,
    help='Ratio of the validation data to the test data'
)
@click.option(
    '--train_batch_size', type=int, default=100, show_default=True,
    help='Batch size for training'
)
@click.option(
    '--val_batch_size', type=int, default=100, show_default=True,
    help='Batch size for validation'
)
@click.option(
    '--test_batch_size', type=int, default=1, show_default=True,
    help='Batch size for testing'
)
@click.option(
    '--loss_threshold', type=float, default=None, show_default=True,
    help='Threshold for the loss function (Stop training if loss goes below this value)'
)
@click.option(
    '--save_interval', type=int, default=0, show_default=True,
    help='Save model weights every N epochs (0 means no saving)'
)
@click.option('-v', '--verbose', help='Verbose output', count=True)
def train(
        data_file: str = "surrogate_data.csv",
        output_dir: str = "output",
        n_epochs: int = 10000,
        patience: int = 500,

        train_test_split_ratio: float = 0.2,
        val_test_split_ratio: float = 0.5,
        train_batch_size: int = 100,
        val_batch_size: int = 100,
        test_batch_size: int = 1,

        loss_threshold: float = None,
        save_interval: int = 0,

        verbose: int = 0,
    ) -> None:
    """Train wood microstructure model"""
    from aitw_monomer_surrogate.train_surrogate import train_surrogate

    train_surrogate(
        n_epochs=n_epochs,
        patience=patience,
        data_file=data_file,
        output_dir=output_dir,

        train_test_split_ratio=train_test_split_ratio,
        val_test_split_ratio=val_test_split_ratio,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,

        loss_threshold=loss_threshold,
        save_interval=save_interval,

        log_level=verbose_map.get(verbose, logging.DEBUG)
    )


__all__ = [
    'train'
]
