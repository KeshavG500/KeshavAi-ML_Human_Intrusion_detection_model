"""
setup_path.py
─────────────
Run this at the top of every notebook to correctly add the
project root to sys.path, regardless of where Jupyter was launched.

Usage in any notebook cell:
    %run setup_path.py
"""
import sys
from pathlib import Path

def _find_project_root() -> Path:
    """
    Walks upward from cwd until it finds a directory containing
    both 'src/' and 'requirements.txt' — that is the project root.
    Falls back to cwd().parent if not found.
    """
    candidates = [Path.cwd()] + list(Path.cwd().parents)
    for path in candidates:
        if (path / "src").is_dir() and (path / "requirements.txt").exists():
            return path
    # Last resort: assume notebooks/ is one level inside the project
    return Path.cwd().parent

ROOT = _find_project_root()

# Ensure it is at position 0 (highest priority)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print(f"✅ Project root : {ROOT}")
print(f"   sys.path[0]  : {sys.path[0]}")
print(f"   src/ exists  : {(ROOT / 'src').is_dir()}")
