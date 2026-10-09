"""AITW Monomer Infiltration Surrogate CLI commands"""
import os
import sys
from importlib import resources

import numpy as np
import pandas as pd

from .main import cli, click


@cli.command()
@click.option(
    '--input_file', type=click.Path(exists=True, dir_okay=False), required=True,
    help='Path to the CSV file containing the input data for inference'
)
@click.option(
    '--weight_file', type=click.Path(exists=True, dir_okay=False),
    help='Path to model weight file'
)
@click.option(
    '--output_file', type=click.Path(), default="inference_results.csv",
    help='Path to the CSV file where the inference results will be saved'
)
def inference(
        input_file: str,
        weight_file: str, 
        output_file: str = "inference_results.csv",
    ) -> None:
    """Run an inference using the trained surrogate model."""
    # Defer heavy imports until this function is called to reduce startup time for the CLI
    import torch
    from sklearn.metrics import r2_score

    from aitw_infiltration_surrogate.model import Net

    if not os.path.exists(input_file):
        click.echo(f"Input file '{input_file}' does not exist.", err=True)
        sys.exit(1)

    if weight_file is None:
        # Use default weight file from package resources
        with resources.path('aitw_infiltration_surrogate', 'weights.pt') as default_weight_path:
            weight_file = str(default_weight_path)
    elif not os.path.exists(weight_file):
        click.echo(f"Weight file '{weight_file}' does not exist.", err=True)
        sys.exit(1)
    else:
        weight_file = os.path.abspath(weight_file)

    click.echo(f"Using weight file: {weight_file}")

    model = Net()
    model.load_state_dict(torch.load(weight_file))

    input_data = pd.read_csv(input_file)

    # Remove the `sat_time` column if it exists, as it's not needed for inference
    validation_data = None
    if 'sat_time' in input_data.columns:
        validation_data = input_data[['sat_time']].copy()
        input_data = input_data.drop(columns=['sat_time'])
        click.secho(
            "`sat_time` column found and removed from input data for inference. "
            "It will be used for computing the RMSE after inference.",
            fg='yellow'
        )  

    # Check that the shape of the input data matches the expected input shape of the model
    if input_data.shape[1] != 6:
        click.echo(
            f"Input data shape {input_data.shape} has more than 6 columns {input_data.columns.tolist()}.", 
            err=True
        )
        sys.exit(1)

    click.echo(f"Running inference on {input_data.shape[0]} samples...")
    # Use the same log transformation as during training
    input_data = input_data.map(np.log)
    input_data = torch.tensor(input_data.values, dtype=torch.float32)
    
    result: torch.Tensor = model(input_data)

    result_df = pd.DataFrame(result.detach().numpy(), columns=['Saturation Time'])

    result_df['Saturation Time'] = np.exp(result_df['Saturation Time'])
    result_df.to_csv(output_file, index=False)

    if validation_data is not None:
        dev_sq = np.sqrt(np.mean((result_df['Saturation Time'] - validation_data['sat_time'])**2))
        click.echo(f"  RMSE against `sat_time` column: {dev_sq:.4f}")
        r2 = r2_score(validation_data['sat_time'], result_df['Saturation Time'])
        click.echo(f"  R^2 score against `sat_time` column: {r2:.4f}")

    click.secho(f"Results saved to {output_file}", fg='green')



__all__ = [
    'inference'
]
