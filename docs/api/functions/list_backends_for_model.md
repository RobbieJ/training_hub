# `list_backends_for_model()` - Preferred Backends By Model

Returns preferred backends for a specific model and algorithm.

## Signature

```python
def list_backends_for_model(
    model_path_or_architecture: str,
    algorithm: str,
    trust_remote_code: bool | None = None,
) -> list[str]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path_or_architecture` | `str` | **Required** | Hugging Face model ID/path or architecture name. |
| `algorithm` | `str` | **Required** | Algorithm name (`"sft"`, `"osft"`, `"lora_sft"`). |
| `trust_remote_code` | `bool \| None` | `None` | Optional hint used when architecture resolution requires remote config code. |

## Returns

Ordered list of preferred backend names for the requested model + algorithm pair.

## Example

```python
from training_hub import list_backends_for_model

backends = list_backends_for_model(
    "Mistral3ForConditionalGeneration",
    algorithm="sft",
)
print(backends)
# ['mini-trainer']
```
