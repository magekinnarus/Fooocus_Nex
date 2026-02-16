import gguf
import argparse
import os

def print_gguf_keys(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Inspecting GGUF: {os.path.basename(file_path)} ---\n")
    
    # Load the GGUF file reader
    reader = gguf.GGUFReader(file_path)

    # 1. Print Metadata (Architecture, Version, etc.)
    print("Metadata:")
    for key in reader.fields:
        field = reader.fields[key]
        # Some values are arrays/lists, handle them for printing
        val = field.parts[field.data[0]] if field.data else "N/A"
        print(f"  {key}: {val}")

    print("\n" + "="*50 + "\n")

    # 2. Print Tensor Keys
    print(f"Tensors Found: {len(reader.tensors)}")
    print("-" * 30)
    for tensor in reader.tensors:
        # tensor.name is the layer key
        # tensor.tensor_type is the quantization type (e.g., Q8_0)
        print(f"Key: {tensor.name: <60} | Type: {tensor.tensor_type.name} | Shape: {tensor.shape}")

    print(f"\nTotal Tensors: {len(reader.tensors)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print layer keys from a GGUF model")
    parser.add_argument("file", help="Path to the .gguf file")
    args = parser.parse_args()

    print_gguf_keys(args.file)