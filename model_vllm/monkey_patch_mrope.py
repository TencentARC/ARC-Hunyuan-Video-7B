import sys
import importlib

# Module where the original MRotaryEmbedding is defined
VLLM_ROTARY_EMBEDDING_MODULE = "vllm.model_executor.layers.rotary_embedding"

# Path to your custom class
# Adjust the import path if your project structure is different
YOUR_CUSTOM_MODULE = "model_vllm.hunyuan"
YOUR_CUSTOM_CLASS_NAME = "DynamicNTKAlphaMRotaryEmbedding"

try:
    # Import the vLLM module
    vllm_rotary_module = importlib.import_module(VLLM_ROTARY_EMBEDDING_MODULE)
    
    # Import your custom class
    custom_module = importlib.import_module(YOUR_CUSTOM_MODULE)
    CustomRotaryEmbeddingClass = getattr(custom_module, YOUR_CUSTOM_CLASS_NAME)
    
    # Perform the monkey patch:
    # Replace the MRotaryEmbedding in the vLLM module with your class
    setattr(vllm_rotary_module, "MRotaryEmbedding", CustomRotaryEmbeddingClass)
    
    print(f"Successfully monkey-patched 'MRotaryEmbedding' in '{VLLM_ROTARY_EMBEDDING_MODULE}' "
          f"with '{YOUR_CUSTOM_CLASS_NAME}' from '{YOUR_CUSTOM_MODULE}'.")
          
except ImportError as e:
    print(f"Error during monkey patching: Could not import modules. {e}")
    print("Please ensure that vLLM is installed and your custom module path is correct.")
except AttributeError as e:
    print(f"Error during monkey patching: Could not find class/attribute. {e}")
    print("Please ensure class names and module contents are correct.")
except Exception as e:
    print(f"An unexpected error occurred during monkey patching: {e}")

