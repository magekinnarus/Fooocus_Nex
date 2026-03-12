import torch
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.upscaler import list_available_models, perform_upscale
import cv2

def test_upscale():
    models = list_available_models()
    print(f"Discovered models: {models}")
    
    if not models:
        print("No models found. Skipping test.")
        return

    # Create dummy image
    img = np.zeros((128, 128, 3), dtype=np.float32)
    cv2.rectangle(img, (20, 20), (100, 100), (0.5, 0.5, 0.5), -1)
    
    for model_name in models:
        print(f"\n--- Testing with {model_name} ---")
        try:
            from modules.upscaler import load_model, get_model_scale
            model = load_model(model_name)
            scale = get_model_scale(model)
            print(f"Detected scale: {scale}x")
            
            result = perform_upscale(img, model_name=model_name)
            print(f"Upscale successful. Output shape: {result.shape}")
            
            # Verify dimensions
            expected_h, expected_w = 128 * scale, 128 * scale
            if result.shape[0] == expected_h and result.shape[1] == expected_w:
                print("Output dimensions are CORRECT.")
            else:
                print(f"Output dimensions are INCORRECT! Expected ({expected_h}, {expected_w}), got {result.shape[:2]}")
                
        except Exception as e:
            print(f"Upscale failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_upscale()
