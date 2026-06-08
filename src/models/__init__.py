"""Model backends for batch inference.

Each submodule exposes two functions with a common signature:

    load(model_path: str, device: str = "cuda") -> Any
        Load model weights and return an opaque bundle.

    predict(bundle, collage, question: str, *, max_new_tokens: int, **kwargs) -> str
        Run one forward pass and return the decoded text response.
"""
