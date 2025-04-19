#!/usr/bin/env python3
import numpy as np
import argparse
import sys
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt

class NPZInspector:
    def __init__(self, file_path):
        """Initialize with path to NPZ file"""
        self.file_path = file_path
        try:
            self.data = np.load(file_path, allow_pickle=True)
            self.keys = sorted(list(self.data.keys()))
            print(f"Successfully loaded NPZ file: {file_path}")
            print(f"Found {len(self.keys)} arrays")
        except Exception as e:
            print(f"Error loading NPZ file: {e}")
            self.data = None
            self.keys = []
    
    def print_structure(self, detailed=False):
        """Print the structure of the NPZ file"""
        if not self.data:
            return
        
        print("\nNPZ File Structure:")
        print("=" * 50)
        
        # Group keys by potential structure
        spc_keys = []
        sdf_keys = []
        other_keys = []
        
        for key in self.keys:
            if 'octree' in key.lower() or 'point' in key.lower() or 'pyramid' in key.lower() or 'spc' in key.lower():
                spc_keys.append(key)
            elif 'weight' in key.lower() or 'bias' in key.lower() or 'sdf' in key.lower() or 'layer' in key.lower():
                sdf_keys.append(key)
            else:
                other_keys.append(key)
        
        def print_array_info(key, arr):
            shape_str = f"shape={arr.shape}" if hasattr(arr, 'shape') else "not an array"
            dtype_str = f"dtype={arr.dtype}" if hasattr(arr, 'dtype') else f"type={type(arr)}"
            size_str = f"size={arr.size}" if hasattr(arr, 'size') else ""
            
            print(f"  {key}: {shape_str}, {dtype_str}, {size_str}")
            
            if detailed and hasattr(arr, 'shape') and len(arr.shape) > 0:
                # For detailed view, print some statistics or sample values
                if np.issubdtype(arr.dtype, np.number):
                    if arr.size > 0:
                        print(f"    Min: {np.min(arr)}, Max: {np.max(arr)}, Mean: {np.mean(arr)}")
                        if arr.size > 10:
                            print(f"    First 5 values: {arr.flatten()[:5]}")
                            print(f"    Last 5 values: {arr.flatten()[-5:]}")
                        else:
                            print(f"    Values: {arr}")
                elif arr.dtype == np.dtype('object'):
                    print(f"    Object array content sample: {arr.flatten()[0] if arr.size > 0 else 'empty'}")
        
        if spc_keys:
            print("\nStructured Point Cloud (SPC) Arrays:")
            for key in sorted(spc_keys):
                print_array_info(key, self.data[key])
        
        if sdf_keys:
            print("\nSigned Distance Function (SDF) Arrays:")
            for key in sorted(sdf_keys):
                print_array_info(key, self.data[key])
        
        if other_keys:
            print("\nOther Arrays:")
            for key in sorted(other_keys):
                print_array_info(key, self.data[key])
    
    def visualize_arrays(self, max_arrays=5):
        """Visualize a few of the arrays for inspection"""
        if not self.data:
            return
        
        # Find arrays suitable for visualization (2D numerical arrays)
        viz_candidates = []
        
        for key in self.keys:
            arr = self.data[key]
            if hasattr(arr, 'shape') and len(arr.shape) >= 1 and np.issubdtype(arr.dtype, np.number):
                viz_candidates.append((key, arr))
        
        if not viz_candidates:
            print("No suitable arrays found for visualization")
            return
        
        # Limit the number of visualizations
        viz_candidates = viz_candidates[:max_arrays]
        
        # Create plots
        num_plots = len(viz_candidates)
        fig_cols = min(2, num_plots)
        fig_rows = (num_plots + fig_cols - 1) // fig_cols
        
        plt.figure(figsize=(15, 5 * fig_rows))
        
        for i, (key, arr) in enumerate(viz_candidates):
            plt.subplot(fig_rows, fig_cols, i + 1)
            
            # Flatten arrays with more than 2 dimensions
            if len(arr.shape) > 2:
                arr = arr.reshape(arr.shape[0], -1)
                title = f"{key} (reshaped from {self.data[key].shape})"
            else:
                title = f"{key} {arr.shape}"
            
            if len(arr.shape) == 1:
                plt.plot(arr)
                plt.grid(True)
            else:  # 2D array
                plt.imshow(arr, cmap='viridis', aspect='auto')
                plt.colorbar()
            
            plt.title(title)
        
        plt.tight_layout()
        plt.show()
    
    def export_summary(self, output_file):
        """Export a JSON summary of the NPZ structure"""
        if not self.data:
            return
        
        summary = {
            "file_path": str(self.file_path),
            "num_arrays": len(self.keys),
            "arrays": {}
        }
        
        for key in self.keys:
            arr = self.data[key]
            if hasattr(arr, 'shape'):
                summary["arrays"][key] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "size": int(arr.size) if hasattr(arr, 'size') else None
                }
            else:
                summary["arrays"][key] = {
                    "type": str(type(arr))
                }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary exported to {output_file}")
    
    def detect_spc_structure(self):
        """Detect if this NPZ has SPC structure needed by the renderer"""
        spc_required = ['octree', 'points', 'pyramid']
        found = [key for key in spc_required if any(req in key.lower() for req in spc_required)]
        
        if found:
            print("\nSPC Structure Detection:")
            print(f"Found potential SPC arrays: {', '.join(found)}")
            if len(found) == len(spc_required):
                print("✓ This NPZ file appears to contain required SPC data")
            else:
                print(f"✗ Missing potential SPC arrays: {', '.join(set(spc_required) - set(found))}")
        else:
            print("\nSPC Structure Detection: No SPC-related arrays found")
    
    def detect_sdf_structure(self):
        """Detect if this NPZ has SDF network weights"""
        # Look for common NN layer patterns
        weight_keys = [key for key in self.keys if 'weight' in key.lower()]
        bias_keys = [key for key in self.keys if 'bias' in key.lower()]
        
        if weight_keys or bias_keys:
            print("\nSDF Structure Detection:")
            print(f"Found {len(weight_keys)} weight arrays and {len(bias_keys)} bias arrays")
            
            # Try to identify network architecture
            layers = []
            for wk in sorted(weight_keys):
                # Try to find matching bias
                base_name = wk.split('.weight')[0]
                bk = f"{base_name}.bias"
                if bk in bias_keys:
                    w_shape = self.data[wk].shape
                    b_shape = self.data[bk].shape
                    layers.append((base_name, w_shape, b_shape))
            
            if layers:
                print("\nPotential Neural Network Architecture:")
                for i, (name, w_shape, b_shape) in enumerate(layers):
                    print(f"  Layer {i+1}: {name}, Weight shape: {w_shape}, Bias shape: {b_shape}")
                
                print("\n✓ This NPZ file appears to contain SDF network weights")
            else:
                print("✗ Could not determine network architecture from weights")
        else:
            print("\nSDF Structure Detection: No neural network weights found")

def main():
    parser = argparse.ArgumentParser(description="Inspect NPZ file structure")
    parser.add_argument("npz_file", help="Path to the NPZ file")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed information for each array")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize some arrays")
    parser.add_argument("--export", "-e", type=str, help="Export summary to JSON file")
    parser.add_argument("--max-viz", type=int, default=5, help="Maximum number of arrays to visualize")
    
    args = parser.parse_args()
    
    file_path = Path(args.npz_file)
    if not file_path.exists():
        print(f"Error: File not found: {args.npz_file}")
        return 1
    
    inspector = NPZInspector(file_path)
    inspector.print_structure(detailed=args.detailed)
    inspector.detect_spc_structure()
    inspector.detect_sdf_structure()
    
    if args.visualize:
        inspector.visualize_arrays(max_arrays=args.max_viz)
    
    if args.export:
        inspector.export_summary(args.export)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())