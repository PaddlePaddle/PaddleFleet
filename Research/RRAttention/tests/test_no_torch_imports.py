from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
IMPORT_TORCH_RE = re.compile(r"^\s*(from|import)\s+torch\b")


def test_new_python_modules_do_not_import_torch():
    paths = [
        ROOT / "rrattn" / "checkpoint_utils.py",
        ROOT / "rrattn" / "rrattention.py",
        ROOT / "rrattn" / "xattention.py",
        ROOT / "rrattn" / "flexprefill.py",
        ROOT / "scripts" / "autotune_rrattn.py",
        ROOT / "scripts" / "convert_hf_to_paddle.py",
        ROOT / "scripts" / "speed_test.py",
        ROOT / "tests" / "test_convert_hf_to_paddle.py",
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(IMPORT_TORCH_RE.search(line) for line in text.splitlines()), path
