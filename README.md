# Synheart Emotion - Python SDK

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-16%2F16%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/synheart-ai/synheart-emotion-python/actions/workflows/ci.yml/badge.svg)](https://github.com/synheart-ai/synheart-emotion-python/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/synheart-ai/synheart-emotion-python/branch/main/graph/badge.svg)](https://codecov.io/gh/synheart-ai/synheart-emotion-python)
[![PyPI version](https://badge.fury.io/py/synheart-emotion.svg)](https://badge.fury.io/py/synheart-emotion)
[![PyPI downloads](https://img.shields.io/pypi/dm/synheart-emotion.svg)](https://pypi.org/project/synheart-emotion/)

On-device emotion inference from biosignals (heart rate and RR intervals) for Python applications.

## Features

- **Privacy-first**: All processing happens on-device
- **Real-time**: <10ms inference latency (ONNX models)
- **Binary emotion states**: Baseline, Stress
- **Sliding window**: 120s window with 60s step (default, configurable)
- **14 HRV Features**: Comprehensive feature extraction (time-domain, frequency-domain, non-linear)
- **ONNX Support**: ExtraTrees models optimized for on-device inference
- **Python 3.8+**: Modern Python with type hints
- **Thread-safe**: Concurrent data ingestion supported
- **HSI Compatible**: Designed for Human State Interface integration

## Installation

### From PyPI (recommended)

```bash
pip install synheart-emotion
```

### From source

```bash
git clone https://github.com/synheart-ai/synheart-emotion-python.git
cd synheart-emotion-python
pip install -e .
```

### With optional ML dependencies

For advanced model loading (scikit-learn, XGBoost):

```bash
pip install synheart-emotion[ml]
```

### Development installation

```bash
pip install synheart-emotion[dev]
```

### Verify Installation

```bash
# Quick verification
python -c "from synheart_emotion import EmotionEngine, EmotionConfig; print('✓ Installation successful')"

# Run tests
pytest tests/

# Run examples
python examples/basic_usage.py
python examples/cli_demo.py --samples 15
```

### Building from Source

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# This creates:
# - dist/synheart_emotion-0.0.1.tar.gz (source distribution)
# - dist/synheart_emotion-0.0.1-py3-none-any.whl (wheel)
```

### Troubleshooting

**Import Error**: Make sure the package is installed with `pip list | grep synheart-emotion`

**Version Conflicts**: Upgrade dependencies with `pip install --upgrade numpy pandas scipy onnxruntime`

**Missing Dependencies**: Install all requirements with `pip install -r requirements.txt`

**ONNX Runtime Issues**: Ensure onnxruntime is installed: `pip install onnxruntime>=1.15.0`

## Quick Start

```python
from datetime import datetime
from synheart_emotion import EmotionConfig, EmotionEngine

# Create engine with default configuration (120s window, 60s step)
config = EmotionConfig()
engine = EmotionEngine(config)

# Push data from wearable
engine.push(
    hr=72.0,
    rr_intervals_ms=[850.0, 820.0, 830.0, 845.0, 825.0],
    timestamp=datetime.now()
)

# Get inference result when ready
results = engine.consume_ready()
for result in results:
    print(f"Emotion: {result.emotion}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Probabilities: {result.probabilities}")
```

## Examples

### Basic Usage

```python
from datetime import datetime
from synheart_emotion import EmotionConfig, EmotionEngine

# Initialize engine
config = EmotionConfig()
engine = EmotionEngine(config)

# Simulate wearable data stream
hr_data = [72.0, 73.5, 71.8, 74.2, 72.5]
rr_data = [
    [850.0, 820.0, 830.0, 845.0, 825.0],
    [855.0, 815.0, 835.0, 840.0, 830.0],
    # ... more data
]

# Push data
for hr, rr_intervals in zip(hr_data, rr_data):
    engine.push(
        hr=hr,
        rr_intervals_ms=rr_intervals,
        timestamp=datetime.now()
    )

# Consume results
results = engine.consume_ready()
if results:
    result = results[0]
    print(f"Emotion: {result.emotion} ({result.confidence:.1%})")
```

See the `examples/` directory for more comprehensive examples:
- `basic_usage.py` - Simple emotion inference
- `custom_config.py` - Custom configuration and logging
- `streaming_data.py` - Continuous data stream simulation

### Custom Configuration

```python
config = EmotionConfig(
    model_id="ExtraTrees_120_60_nozipmap",  # ExtraTrees model
    window_seconds=120.0,     # 120 second window (default)
    step_seconds=60.0,        # 60 second step (default)
    min_rr_count=30,          # Minimum RR intervals
)
```

### Logging

```python
def custom_logger(level, message):
    print(f"[{level}] {message}")

engine = EmotionEngine(
    config=config,
    on_log=custom_logger
)
```

### Buffer Management

```python
# Get buffer statistics
stats = engine.get_buffer_stats()
print(f"Data points: {stats['count']}")
print(f"Duration: {stats['duration_ms']}ms")
print(f"HR range: {stats['hr_range']}")
print(f"RR count: {stats['rr_count']}")

# Clear buffer
engine.clear()
```

## API Reference

### EmotionConfig

Configuration for the emotion inference engine.

```python
@dataclass
class EmotionConfig:
    model_id: str = "ExtraTrees_120_60_nozipmap"
    window_seconds: float = 120.0
    step_seconds: float = 60.0
    min_rr_count: int = 30
```

**Attributes:**

- `model_id` - Model identifier (default: ExtraTrees_120_60_nozipmap)
- `window_seconds` - Rolling window size (default: 120s)
- `step_seconds` - Emission cadence (default: 60s)
- `min_rr_count` - Minimum RR intervals required (default: 30)

### EmotionEngine

Main emotion inference engine.

**Class Methods:**

```python
def __init__(
    config: EmotionConfig,
    on_log: Optional[Callable[[str, str], None]] = None
) -> EmotionEngine
```

Create engine. The model is automatically loaded based on `config.model_id`.

**Instance Methods:**

```python
def push(
    hr: float,
    rr_intervals_ms: List[float],
    timestamp: datetime,
    motion: Optional[Dict[str, float]] = None
) -> None
```

Push new data point into the engine.

```python
def consume_ready() -> List[EmotionResult]
```

Consume ready results (throttled by step interval).

```python
def get_buffer_stats() -> Dict[str, Any]
```

Get current buffer statistics.

```python
def clear() -> None
```

Clear all buffered data.

### EmotionResult

Result of emotion inference (dictionary).

```python
{
    "timestamp": datetime,
    "emotion": str,              # Top-1 predicted label (Baseline or Stress)
    "confidence": float,         # Confidence score (0.0-1.0)
    "probabilities": Dict[str, float],  # All label probabilities
    "features": Dict[str, float]  # Extracted 14 HRV features
}
```


## Running Examples

```bash
# Basic usage
python examples/basic_usage.py

# Custom configuration
python examples/custom_config.py

# Streaming data simulation
python examples/streaming_data.py
```

## Requirements

- Python 3.8+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0 (for frequency-domain HRV features)
- onnxruntime >= 1.15.0 (for ONNX model inference)

Optional (for ML model loading):
- scikit-learn >= 1.0.0
- joblib >= 1.1.0
- xgboost >= 1.5.0

## Architecture

The package follows a modular architecture:

```
synheart_emotion/
├── __init__.py          # Package exports
├── synheart_emotion.py  # Single-file implementation (config, engine, features, ONNX)
└── data/                # ONNX model files and metadata
```

### Data Flow

1. **Push** - Biosignal data (HR, RR intervals) pushed to engine
2. **Buffer** - Data stored in sliding window ring buffer
3. **Window Check** - Engine verifies window is full (oldest data >= window_seconds)
4. **Extract** - 14 HRV features extracted from window data (time-domain, frequency-domain, non-linear)
5. **Infer** - ONNX model predicts emotion probabilities
6. **Emit** - Results emitted at configured step intervals

### Thread Safety

The engine uses `threading.RLock()` for thread-safe operations:
- Multiple threads can push data concurrently
- Buffer operations are protected
- Results can be consumed from any thread

## Model Architecture

The library uses **ExtraTrees (Extremely Randomized Trees)** classifiers trained on the WESAD dataset:

- **14 HRV Features**: Time-domain, frequency-domain, and non-linear metrics
- **Binary Classification**: Baseline vs Stress detection
- **ONNX Format**: Optimized for on-device inference using ONNX Runtime
- **Accuracy**: ~78% on WESAD validation set

### Available Models

Models are automatically loaded based on `config.model_id`:

- `extratrees_w120s60_binary_v1_0` or `ExtraTrees_120_60_nozipmap`: 120-second window, 60-second step (default)
- `extratrees_w60s5_binary_v1_0` or `ExtraTrees_60_5_nozipmap`: 60-second window, 5-second step
- `extratrees_w120s5_binary_v1_0` or `ExtraTrees_120_5_nozipmap`: 120-second window, 5-second step

All models use binary classification: **Baseline** vs **Stress**.

### Feature Extraction

The library extracts 14 HRV features in the following order:

**Time-domain features:**
- RMSSD (Root Mean Square of Successive Differences)
- Mean_RR (Mean RR interval)
- HRV_SDNN (Standard Deviation of NN intervals)
- pNN50 (Percentage of successive differences > 50ms)

**Frequency-domain features:**
- HRV_HF (High Frequency power)
- HRV_LF (Low Frequency power)
- HRV_HF_nu (Normalized HF)
- HRV_LF_nu (Normalized LF)
- HRV_LFHF (LF/HF ratio)
- HRV_TP (Total Power)

**Non-linear features:**
- HRV_SD1SD2 (Poincaré plot ratio)
- HRV_Sampen (Sample Entropy)
- HRV_DFA_alpha1 (Detrended Fluctuation Analysis)

**Heart Rate:**
- HR (Heart Rate in BPM)

## Privacy & Security

- **On-Device Processing**: All emotion inference happens locally
- **No Data Retention**: Raw biometric data is not retained after processing
- **No Network Calls**: No data is sent to external servers
- **Privacy-First Design**: No built-in storage - you control what gets persisted
- **Real Trained Models**: Uses WESAD-trained ExtraTrees models with ~78% accuracy
- **14-Feature Extraction**: Comprehensive HRV analysis including time-domain, frequency-domain, and non-linear metrics

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ examples/ tests/
isort src/ examples/ tests/
```

### Type Checking

```bash
mypy src/
```

## Integration

### With synheart-core (HSI)

**synheart_emotion** is designed to integrate seamlessly with [synheart-core](https://github.com/synheart-ai/synheart-core) as part of the Human State Interface (HSI) system:

```python
from synheart_core import Synheart, SynheartConfig
from synheart_emotion import EmotionEngine, EmotionConfig

# Initialize synheart-core (includes emotion capability)
synheart = Synheart.initialize(
    user_id="user_123",
    config=SynheartConfig(
        enable_wear=True,
        enable_behavior=True,
    ),
)

# Enable emotion interpretation layer (powered by synheart-emotion)
synheart.enable_emotion()

# Get emotion updates through HSI
@synheart.on_emotion_update
def handle_emotion(emotion):
    print(f"Baseline: {emotion.baseline}")
    print(f"Stress: {emotion.stress}")
```

**HSI Schema Compatibility:**
- EmotionResult from synheart-emotion maps to HSI EmotionState
- Output validated against HSI_SPECIFICATION.md
- Comprehensive integration tests ensure compatibility

See the [synheart-core documentation](https://github.com/synheart-ai/synheart-core) for more details on HSI integration.

## Performance

**Target Performance:**
- **Latency**: < 10ms per inference (ONNX models)
- **Model Size**: ~200-300 KB per model
- **CPU Usage**: < 3% during active streaming
- **Memory**: < 5 MB (engine + buffers + ONNX runtime)
- **Accuracy**: ~78% on WESAD dataset (binary classification: Baseline vs Stress)

**Benchmarks:**
- 14-feature extraction: < 3ms
- ONNX model inference: < 5ms
- Full pipeline: < 10ms

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! See our [Contributing Guidelines](https://github.com/synheart-ai/synheart-emotion/blob/main/CONTRIBUTING.md) for details.

## 🔗 Links

- **Main Repository**: [synheart-emotion](https://github.com/synheart-ai/synheart-emotion) (Source of Truth)
- **Documentation**: [RFC E1.1](https://github.com/synheart-ai/synheart-emotion/blob/main/docs/RFC-E1.1.md)
- **Model Card**: [Model Card](https://github.com/synheart-ai/synheart-emotion/blob/main/docs/MODEL_CARD.md)
- **Examples**: [Examples](https://github.com/synheart-ai/synheart-emotion/tree/main/examples)
- **Models**: [Pre-trained Models](https://github.com/synheart-ai/synheart-emotion/tree/main/models)
- **Tools**: [Development Tools](https://github.com/synheart-ai/synheart-emotion/tree/main/tools)
- **Synheart AI**: [synheart.ai](https://synheart.ai)
- **Issues**: [GitHub Issues](https://github.com/synheart-ai/synheart-emotion-python/issues)

## Citation

If you use this package in your research, please cite:

```bibtex
@software{synheart_emotion,
  title = {Synheart Emotion: On-device emotion inference from biosignals},
  author = {Goytom, Israel},
  year = {2025},
  version = {0.0.1},
  url = {https://github.com/synheart-ai/synheart-emotion}
}
```
