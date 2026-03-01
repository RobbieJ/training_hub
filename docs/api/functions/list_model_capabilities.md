# `list_model_capabilities()` - Model Capability Introspection

Returns capability metadata for a model architecture or model path.

## Signature

```python
def list_model_capabilities(
    model_path_or_architecture: str,
    algorithm: str | None = None,
    trust_remote_code: bool | None = None,
) -> dict[str, Any]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path_or_architecture` | `str` | **Required** | Hugging Face model ID/path or architecture name (e.g. `"Mistral3ForConditionalGeneration"`). |
| `algorithm` | `str \| None` | `None` | Optional algorithm filter (`"sft"`, `"osft"`, `"lora_sft"`). |
| `trust_remote_code` | `bool \| None` | `None` | Optional hint used when architecture resolution requires remote config code. |

## Returns

A dictionary with resolved architecture, capability status, backend preferences, and compatibility metadata.

## Example

```python
from training_hub import list_model_capabilities

caps = list_model_capabilities(
    "mistralai/Ministral-3-3B-Instruct-2512",
    algorithm="sft",
    trust_remote_code=True,
)
print(caps)
```
