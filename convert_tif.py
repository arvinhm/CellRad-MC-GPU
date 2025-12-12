#!/usr/bin/env python3
"""
Convert multi-channel TIF microscopy images to NIfTI format for CellRad-MC simulation.

Supports:
- Multi-channel (e.g., DAPI, EGFR, CD31) TIF files
- Z-stack removal of out-of-focus slices
- Automatic voxel size extraction from bioformat metadata
- Output as separate NIfTI files per channel

Usage:
    python convert_tif.py --file image.tif --channels 3 --remove 1,25,41
    python convert_tif.py --file image.tif --channels 3 --channel-names DAPI,EGFR,CD31
"""

import argparse
import numpy as np
import tifffile
import nibabel as nib
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert multi-channel TIF to NIfTI for CellRad-MC simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with 3 channels
  python convert_tif.py --file image.tif --channels 3

  # Remove out-of-focus slices (1-indexed)
  python convert_tif.py --file image.tif --channels 3 --remove 1,25,41,52,54,55,56,57,58

  # Specify channel names for output files
  python convert_tif.py --file image.tif --channels 3 --channel-names DAPI,EGFR,CD31

  # Specify output directory
  python convert_tif.py --file image.tif --channels 3 --out-dir converted/
        """
    )
    
    parser.add_argument('--file', type=str, required=True,
                        help='Input TIF file path')
    parser.add_argument('--channels', type=int, required=True,
                        help='Number of channels in the TIF file')
    parser.add_argument('--remove', type=str, default='',
                        help='Comma-separated list of slice numbers to remove (1-indexed)')
    parser.add_argument('--channel-names', type=str, default='',
                        help='Comma-separated channel names (e.g., DAPI,EGFR,CD31)')
    parser.add_argument('--out-dir', type=str, default='',
                        help='Output directory (default: same as input file)')
    
    return parser.parse_args()


def extract_voxel_size(tif):
    """
    Extract voxel size from TIF metadata (bioformat/ImageJ).
    
    Returns:
        tuple: (x_size, y_size, z_size) in microns
    """
    x_size = 1.0
    y_size = 1.0
    z_size = 1.0
    unit = 'unknown'
    
    # Try ImageJ metadata first
    if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
        meta = tif.imagej_metadata
        
        # Z spacing
        if 'spacing' in meta:
            z_size = float(meta['spacing'])
        
        # Unit
        if 'unit' in meta:
            unit = meta['unit']
    
    # Get XY resolution from TIFF tags
    page = tif.pages[0]
    if 'XResolution' in page.tags:
        x_res_tag = page.tags['XResolution'].value
        if isinstance(x_res_tag, tuple) and len(x_res_tag) == 2:
            x_res = x_res_tag[0] / x_res_tag[1]  # pixels per unit
            x_size = 1.0 / x_res if x_res > 0 else 1.0
    
    if 'YResolution' in page.tags:
        y_res_tag = page.tags['YResolution'].value
        if isinstance(y_res_tag, tuple) and len(y_res_tag) == 2:
            y_res = y_res_tag[0] / y_res_tag[1]  # pixels per unit
            y_size = 1.0 / y_res if y_res > 0 else 1.0
    
    return x_size, y_size, z_size, unit


def load_tif(file_path):
    """
    Load TIF file and return data with metadata.
    
    Returns:
        tuple: (data, axes, voxel_size, unit, metadata)
    """
    print(f"\nLoading: {file_path}")
    
    with tifffile.TiffFile(file_path) as tif:
        # Get data
        data = tif.asarray()
        
        # Get axes
        if len(tif.series) > 0:
            axes = tif.series[0].axes
        else:
            axes = 'Unknown'
        
        # Get voxel size
        x_size, y_size, z_size, unit = extract_voxel_size(tif)
        
        # Get additional metadata
        metadata = {}
        if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
            metadata = tif.imagej_metadata.copy()
    
    return data, axes, (x_size, y_size, z_size), unit, metadata


def save_nifti(data, voxel_size, output_path):
    """
    Save 3D numpy array as NIfTI file.
    
    Args:
        data: 3D numpy array (X, Y, Z)
        voxel_size: tuple (x, y, z) in microns
        output_path: output file path
    """
    # Create affine matrix with voxel sizes
    affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])
    
    # Convert to float32 for simulation compatibility
    data_float = data.astype(np.float32)
    
    # Create NIfTI image
    nii = nib.Nifti1Image(data_float, affine)
    
    # Set header info
    nii.header.set_xyzt_units('micron')
    
    # Save
    nib.save(nii, output_path)
    print(f"  Saved: {output_path}")
    print(f"    Shape: {data.shape}")
    print(f"    Voxel size: {voxel_size[0]:.4f} x {voxel_size[1]:.4f} x {voxel_size[2]:.4f} um")
    print(f"    Value range: {data.min():.1f} - {data.max():.1f}")


def main():
    args = parse_args()
    
    # Check input file exists
    input_path = Path(args.file)
    if not input_path.exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    # Set output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = input_path.parent
    
    # Parse slices to remove (convert from 1-indexed to 0-indexed)
    remove_slices = []
    if args.remove:
        try:
            remove_slices = [int(s.strip()) - 1 for s in args.remove.split(',') if s.strip()]
            print(f"\nSlices to remove (1-indexed): {args.remove}")
            print(f"Slices to remove (0-indexed): {remove_slices}")
        except ValueError:
            print(f"Error: Invalid slice numbers: {args.remove}")
            sys.exit(1)
    
    # Parse channel names
    if args.channel_names:
        channel_names = [n.strip() for n in args.channel_names.split(',')]
        if len(channel_names) != args.channels:
            print(f"Warning: Number of channel names ({len(channel_names)}) != channels ({args.channels})")
            channel_names = [f'channel_{i+1}' for i in range(args.channels)]
    else:
        channel_names = [f'channel_{i+1}' for i in range(args.channels)]
    
    # Load TIF
    data, axes, voxel_size, unit, metadata = load_tif(args.file)
    
    print(f"\n{'='*60}")
    print("TIF File Information")
    print('='*60)
    print(f"  Shape: {data.shape}")
    print(f"  Axes: {axes}")
    print(f"  Data type: {data.dtype}")
    print(f"  Channels: {args.channels}")
    print(f"  Voxel size: {voxel_size[0]:.4f} x {voxel_size[1]:.4f} x {voxel_size[2]:.4f} {unit}")
    
    if metadata:
        print(f"\n  ImageJ Metadata:")
        for key, val in metadata.items():
            print(f"    {key}: {val}")
    
    # Determine data layout based on axes
    # Common formats: ZCYX, CZYX, TZCYX, etc.
    print(f"\n{'='*60}")
    print("Processing Data")
    print('='*60)
    
    if axes == 'ZCYX':
        # Shape: (Z, C, Y, X)
        n_slices = data.shape[0]
        n_channels = data.shape[1]
        print(f"  Detected format: ZCYX")
        print(f"  Z slices: {n_slices}")
        print(f"  Channels: {n_channels}")
        
        if n_channels != args.channels:
            print(f"  Warning: Detected {n_channels} channels, but --channels={args.channels}")
        
        # Create list of slices to keep
        keep_slices = [i for i in range(n_slices) if i not in remove_slices]
        print(f"  Keeping {len(keep_slices)} of {n_slices} slices")
        
        # Extract and save each channel
        print(f"\n{'='*60}")
        print("Saving NIfTI Files")
        print('='*60)
        
        for c in range(min(n_channels, args.channels)):
            # Extract channel data: (Z, Y, X)
            channel_data = data[keep_slices, c, :, :]
            
            # Transpose to (X, Y, Z) for NIfTI convention
            channel_data = np.transpose(channel_data, (2, 1, 0))
            
            # Generate output filename
            base_name = input_path.stem.replace(' ', '_').replace('-', '_')
            output_name = f"{base_name}_{channel_names[c]}.nii.gz"
            output_path = out_dir / output_name
            
            print(f"\nChannel {c+1}: {channel_names[c]}")
            save_nifti(channel_data, voxel_size, str(output_path))
    
    elif axes == 'CZYX':
        # Shape: (C, Z, Y, X)
        n_channels = data.shape[0]
        n_slices = data.shape[1]
        print(f"  Detected format: CZYX")
        print(f"  Channels: {n_channels}")
        print(f"  Z slices: {n_slices}")
        
        # Create list of slices to keep
        keep_slices = [i for i in range(n_slices) if i not in remove_slices]
        print(f"  Keeping {len(keep_slices)} of {n_slices} slices")
        
        # Extract and save each channel
        print(f"\n{'='*60}")
        print("Saving NIfTI Files")
        print('='*60)
        
        for c in range(min(n_channels, args.channels)):
            # Extract channel data: (Z, Y, X)
            channel_data = data[c, keep_slices, :, :]
            
            # Transpose to (X, Y, Z) for NIfTI convention
            channel_data = np.transpose(channel_data, (2, 1, 0))
            
            # Generate output filename
            base_name = input_path.stem.replace(' ', '_').replace('-', '_')
            output_name = f"{base_name}_{channel_names[c]}.nii.gz"
            output_path = out_dir / output_name
            
            print(f"\nChannel {c+1}: {channel_names[c]}")
            save_nifti(channel_data, voxel_size, str(output_path))
    
    else:
        # Try to handle unknown format
        print(f"  Unknown axes format: {axes}")
        print(f"  Attempting automatic detection...")
        
        # Assume last two dimensions are Y, X
        # Look for channel dimension (usually 3 or small number)
        shape = data.shape
        
        if len(shape) == 4:
            # Could be (Z, C, Y, X) or (C, Z, Y, X)
            if shape[1] == args.channels:
                # Likely ZCYX
                n_slices = shape[0]
                n_channels = shape[1]
                print(f"  Guessing format: ZCYX")
                
                keep_slices = [i for i in range(n_slices) if i not in remove_slices]
                
                for c in range(n_channels):
                    channel_data = data[keep_slices, c, :, :]
                    channel_data = np.transpose(channel_data, (2, 1, 0))
                    
                    base_name = input_path.stem.replace(' ', '_').replace('-', '_')
                    output_name = f"{base_name}_{channel_names[c]}.nii.gz"
                    output_path = out_dir / output_name
                    
                    print(f"\nChannel {c+1}: {channel_names[c]}")
                    save_nifti(channel_data, voxel_size, str(output_path))
            
            elif shape[0] == args.channels:
                # Likely CZYX
                n_channels = shape[0]
                n_slices = shape[1]
                print(f"  Guessing format: CZYX")
                
                keep_slices = [i for i in range(n_slices) if i not in remove_slices]
                
                for c in range(n_channels):
                    channel_data = data[c, keep_slices, :, :]
                    channel_data = np.transpose(channel_data, (2, 1, 0))
                    
                    base_name = input_path.stem.replace(' ', '_').replace('-', '_')
                    output_name = f"{base_name}_{channel_names[c]}.nii.gz"
                    output_path = out_dir / output_name
                    
                    print(f"\nChannel {c+1}: {channel_names[c]}")
                    save_nifti(channel_data, voxel_size, str(output_path))
        else:
            print(f"  Error: Cannot handle {len(shape)}D data with shape {shape}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print('='*60)
    print(f"\nOutput directory: {out_dir}")
    print(f"\nVoxel size for simulation:")
    print(f"  X: {voxel_size[0]:.4f} {unit}")
    print(f"  Y: {voxel_size[1]:.4f} {unit}")
    print(f"  Z: {voxel_size[2]:.4f} {unit}")
    print(f"\nTo run simulation, use:")
    if abs(voxel_size[0] - voxel_size[2]) < 0.01:
        # Isotropic voxels
        print(f"  python cellrad_mc.py --protein <channel>.nii.gz --auto-ct --isotope Lu-177 --voxel-size {voxel_size[0]:.2f}")
    else:
        # Anisotropic voxels
        print(f"  python cellrad_mc.py --protein <channel>.nii.gz --auto-ct --isotope Lu-177 --voxel-size {voxel_size[0]:.4f} --voxel-size-z {voxel_size[2]:.4f}")


if __name__ == '__main__':
    main()

