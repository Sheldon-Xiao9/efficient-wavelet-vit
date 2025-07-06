import torch
import argparse
from thop import profile # type: ignore
from network.model import DeepfakeDetector

def calculate_gflops(model, input_tensor, batch_size, ablation):
    """Calculates and prints the GFLOPS of the model."""
    # Set the model to evaluation mode
    model.eval()

    # Profile the model
    flops, params = profile(model, inputs=(input_tensor, batch_size, ablation))

    # Convert FLOPs to GFLOPS
    gflops = flops / 1e9

    print(f"Ablation: {ablation}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"GFLOPS: {gflops:.4f}")
    print(f"Parameters (M): {params / 1e6:.4f}")
    print("-" * 30)

def main():
    """Main function to calculate GFLOPS based on command-line arguments."""
    parser = argparse.ArgumentParser(description="Calculate GFLOPS for different model ablations.")
    parser.add_argument(
        '--ablation',
        type=str,
        default='all',
        choices=['dynamic', 'sfe_only', 'sfe_mwt', 'all'],
        help="Specify the ablation setting to calculate GFLOPS for. Choose from 'dynamic', 'sfe_only', 'sfe_mwt', or 'all'."
    )
    args = parser.parse_args()

    # Common parameters
    batch_size = 1
    num_frames = 24
    in_channels = 3
    height = 224
    width = 224
    dama_dim = 128

    # Create a dummy input tensor
    dummy_input = torch.randn(batch_size, num_frames, in_channels, height, width)

    def run_calculation(ablation_type):
        """Helper function to create model and calculate GFLOPS."""
        model = DeepfakeDetector(
            in_channels=in_channels,
            dama_dim=dama_dim,
            batch_size=batch_size,
            ablation=ablation_type
        )
        calculate_gflops(model, dummy_input, batch_size, ablation_type)

    if args.ablation == 'all':
        run_calculation('dynamic')
        run_calculation('sfe_only')
        run_calculation('sfe_mwt')
    else:
        run_calculation(args.ablation)

if __name__ == '__main__':
    main()
