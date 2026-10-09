"""Test the CLI of the aitw-infiltration-surrogate package."""
import os

from click.testing import CliRunner


def test_cli_train_and_inference(dataset_file_path, tmpdir):
    """Test the CLI train command."""
    from aitw_infiltration_surrogate.cli import cli

    tmp_output_dir = tmpdir / "output"

    runner = CliRunner()
    result = runner.invoke(cli, [
        'train',
        '--n_epochs', '1',
        '--data_file', dataset_file_path,
        '--output_dir', str(tmp_output_dir),
    ])

    assert result.exit_code == 0
    assert os.path.exists(str(tmp_output_dir / 'weights.pt'))

    result = runner.invoke(cli, [
        'inference',
        '--input_file', dataset_file_path,
        '--weight_file', str(tmp_output_dir / 'weights.pt'),
        '--output_file', str(tmpdir / 'inference_results.csv'),
    ])

    assert result.exit_code == 0
    assert os.path.exists(str(tmpdir / 'inference_results.csv'))

def test_cli_inference_with_default_weights(dataset_file_path, tmpdir):
    """Test the CLI inference command with default weights."""
    from aitw_infiltration_surrogate.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        'inference',
        '--input_file', dataset_file_path,
        '--output_file', str(tmpdir / 'inference_results.csv'),
    ])

    assert result.exit_code == 0
    assert os.path.exists(str(tmpdir / 'inference_results.csv'))
