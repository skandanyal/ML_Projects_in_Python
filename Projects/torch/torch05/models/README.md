## Models Directory

The trained model weights are saved in the project's `models/` directory. Before running the training notebook or script, create a `models/` folder in the project root.

### Directory Structure

```text
project-root/
├── data/
├── models/
...
```

### Saving Trained Models

The training code saves the model `state_dict` files to the `models/` directory:

```python
MODEL_DIR = BASE_DIR / "models"

cnn_path = MODEL_DIR / "torch05_cnn_skip_state_dict.pth"
rn_path = MODEL_DIR / "torch05_rn_state_dict.pth"

torch.save(rn_model.state_dict(), cnn_path)
torch.save(cnn3_rn.state_dict(), rn_path)
```

If the `models/` directory does not exist, Python will raise a `FileNotFoundError`. Create the directory before running the notebook, or use:

```bash
mkdir models
```

or in Python:

```python
from pathlib import Path

(Path(BASE_DIR) / "models").mkdir(exist_ok=True)
```
