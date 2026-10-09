"""AITW Monomer Surrogate CLI commands"""
import os
import sys

from .main import cli, click


@cli.command()
@click.option(
    '--weight_file', type=click.Path(exists=True, dir_okay=False), required=True,
    help='Path to model weight file'
)
def run(
        weight_file: str, 
    ) -> None:
    """Run an inference using the trained surrogate model."""
    import torch

    from aitw_monomer_surrogate.model import Net

    if not os.path.exists(weight_file):
        click.echo(f"Weight file '{weight_file}' does not exist.", err=True)
        sys.exit(1)

    model = Net()
    model.load_state_dict(torch.load(weight_file))
    
    raise NotImplementedError("This command is not yet implemented.")
    result = model(SOME_INPUT)  # TODO: Replace SOME_INPUT with actual input data

    # TODO save/print the result as needed


__all__ = [
    'run'
]
