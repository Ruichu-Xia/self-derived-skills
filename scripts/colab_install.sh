#!/usr/bin/env bash
# Install the training stack on a Colab GPU runtime. Run from the repository root.
set -euo pipefail

pip install -q -e ".[train,cuda]"
# Colab preinstalls a torchao too old for PEFT, which then crashes. This project does not use it.
pip uninstall -q -y torchao

python - <<'PY'
import fla  # noqa: F401
import peft
import torch
import transformers
import trl
from peft.import_utils import is_torchao_available

assert torch.cuda.is_available(), "PyTorch cannot see a GPU"
assert not is_torchao_available(), "torchao is still installed"
print(f"torch {torch.__version__}, transformers {transformers.__version__}, trl {trl.__version__}, peft {peft.__version__}")
PY
