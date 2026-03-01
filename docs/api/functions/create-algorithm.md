# `create_algorithm()` - Algorithm Factory Function

> Factory function to create algorithm instances with a specified backend. Useful for advanced use cases requiring direct access to algorithm classes.

## Signature

```python
from training_hub import create_algorithm

algorithm = create_algorithm(
    algorithm_name: str,
    backend_name: str = None,
    model_path_or_architecture: str | None = None,
    trust_remote_code: bool | None = None,
    **kwargs
) -> Algorithm
```

## Quick Example

```python
from training_hub import create_algorithm

# Create an SFT algorithm with InstructLab backend
algorithm = create_algorithm("sft", backend_name="instructlab-training")

# Use the algorithm
result = algorithm.train(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints"
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm_name` | `str` | **Required** | Name of the algorithm to create. Valid values: `"sft"`, `"osft"`. |
| `backend_name` | `str` | `None` | Name of the backend to use. Supports `"auto"` for model-aware backend routing. |
| `model_path_or_architecture` | `str \| None` | `None` | Optional model reference used by auto-routing and compatibility checks. |
| `trust_remote_code` | `bool \| None` | `None` | Optional compatibility flag passed to capability routing rules. |
| `**kwargs` | `Any` | - | Additional keyword arguments passed to the algorithm's constructor. |

## Returns

**Type:** `Algorithm`

Returns an instance of the requested algorithm class (e.g., `SFTAlgorithm`, `OSFTAlgorithm`).

## Raises

- **`ValueError`**: If the specified `algorithm_name` is not found in the registry.
- **`ValueError`**: If no backends are available for the algorithm.
- **`ValueError`**: If the specified `backend_name` is not found for the algorithm.

## Examples

### Creating an SFT Algorithm

```python
from training_hub import create_algorithm

# Create with default backend
sft_algo = create_algorithm("sft")

# Train
result = sft_algo.train(
    model_path="./model",
    data_path="./data.jsonl",
    ckpt_output_dir="./checkpoints",
    num_epochs=3
)
```

### Creating an OSFT Algorithm with Specific Backend

```python
from training_hub import create_algorithm

# Create OSFT with mini-trainer backend explicitly
osft_algo = create_algorithm("osft", backend_name="mini-trainer")

# Train
result = osft_algo.train(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    data_path="./domain_data.jsonl",
    ckpt_output_dir="./checkpoints",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=1024,
    learning_rate=5e-6
)
```

### Checking Available Algorithms and Backends

```python
from training_hub import AlgorithmRegistry

# List all available algorithms
algorithms = AlgorithmRegistry.list_algorithms()
print(f"Available algorithms: {algorithms}")
# Output: Available algorithms: ['sft', 'osft']

# List backends for a specific algorithm
sft_backends = AlgorithmRegistry.list_backends("sft")
print(f"SFT backends: {sft_backends}")
# Output: SFT backends: ['instructlab-training']

osft_backends = AlgorithmRegistry.list_backends("osft")
print(f"OSFT backends: {osft_backends}")
# Output: OSFT backends: ['mini-trainer']
```

### Using the Factory in a Generic Training Function

```python
from training_hub import create_algorithm

def train_model(algo_name: str, model_path: str, data_path: str, output_dir: str, **train_kwargs):
    """Generic training function that works with any algorithm."""
    algorithm = create_algorithm(algo_name)

    return algorithm.train(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=output_dir,
        **train_kwargs
    )

# Use it for SFT
sft_result = train_model(
    "sft",
    model_path="./model",
    data_path="./data.jsonl",
    output_dir="./sft_output",
    num_epochs=3,
    learning_rate=1e-5
)

# Use it for OSFT
osft_result = train_model(
    "osft",
    model_path="./model",
    data_path="./data.jsonl",
    output_dir="./osft_output",
    unfreeze_rank_ratio=0.25,
    effective_batch_size=16,
    max_tokens_per_gpu=2048,
    max_seq_len=1024,
    learning_rate=5e-6
)
```

## When to Use This Function

### Use `create_algorithm()` when:

- You need to dynamically select algorithms based on runtime conditions
- You're building a training pipeline that supports multiple algorithm types
- You want to inspect algorithm parameters before training
- You need to reuse an algorithm instance for multiple training runs

### Use convenience functions (`sft()`, `osft()`) when:

- You know which algorithm you want to use at write-time
- You want the simplest, most concise API
- You're doing one-off training runs

## Implementation Notes

?> **Tip**: For most use cases, the convenience functions [`sft()`](/api/functions/sft) and [`osft()`](/api/functions/osft) are simpler and more direct.

> **Note**: This function queries the [`AlgorithmRegistry`](/api/classes/AlgorithmRegistry) to find and instantiate the requested algorithm and backend.

> **Note**: If you don't specify a `backend_name`, the function will use the first registered backend for that algorithm, which is typically the recommended default.

## See Also

- [**AlgorithmRegistry Class**](/api/classes/AlgorithmRegistry) - Registry system for algorithms and backends
- [**Algorithm Class**](/api/classes/Algorithm) - Base class for all algorithms
- [**sft() Function**](/api/functions/sft) - Convenience function for supervised fine-tuning
- [**osft() Function**](/api/functions/osft) - Convenience function for OSFT
- [**Extending the Framework Guide**](/guides/extending-framework) - Creating custom algorithms

## Source

[View source on GitHub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/__init__.py)
