#!/usr/bin/env python3
"""
Generate table of contents for README.md
Usage: python scripts/generate_toc.py [--preview]
"""

import subprocess
import sys
from pathlib import Path

def generate_toc(preview=False):
    """Generate table of contents for README.md"""
    project_root = Path(__file__).parent.parent
    readme_path = project_root / "README.md"
    
    if not readme_path.exists():
        print("Error: README.md not found in project root")
        return 1
    
    # Build command
    cmd = ["uv", "run", "mdformat", "--wrap=no", "--end-of-line=keep"]
    
    if preview:
        cmd.append("--check")
        print("Previewing table of contents...")
    else:
        print("Generating table of contents for README.md...")
    
    cmd.append(str(readme_path))
    
    # Run command
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        if not preview:
            print("✅ Table of contents updated in README.md")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating TOC: {e}")
        return 1

if __name__ == "__main__":
    preview = "--preview" in sys.argv or "--check" in sys.argv
    sys.exit(generate_toc(preview))