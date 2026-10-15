"""The CLI index must also select the corpus used by runtime imports."""
import os
from pathlib import Path
import subprocess
import sys


def test_benchmark_binds_cli_index_before_importing_runtime(tmp_path):
    selected = tmp_path / "selected-index"
    selected.mkdir()
    output = tmp_path / "preflight.json"
    script = r'''
import sys
from pathlib import Path
import visual_bench
visual_bench.load_dataset = lambda *a: {
    'review_status': 'agent_inspected_draft',
    'questions': [{'id': 'q', 'split': 'test'}]}
visual_bench.preflight = lambda *a: {}
visual_bench.provenance = lambda *a: {}
sys.argv = ['visual_bench', '--preflight', '--index', sys.argv[1], '--output', sys.argv[2]]
visual_bench.main()
import corpus
assert corpus.LAYOUT.index_dir == Path(sys.argv[3]).resolve()
'''
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "scripts"),
           "PIXELRAG_INDEX_DIR": str(tmp_path / "old-index")}
    result = subprocess.run([sys.executable, '-B', '-c', script, str(selected), str(output)],
                            cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
