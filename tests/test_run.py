"""Stage bookkeeping in nwrd.run that the end-to-end tests do not isolate."""

import json

from nwrd.run import main


def test_benchmark_is_remeasured_when_a_model_is_missing(tmp_path):
    """A model added to a finished experiment re-measures the whole table, so every row comes
    from one session on one device; a complete table is left alone."""
    out = tmp_path / 'runs'
    out.mkdir()
    bench = out / 'benchmark.json'
    bench.write_text(json.dumps({'env': {'device': 'old session'}, 'results': [{'model': 'unet'}]}))
    args = ['--stage', 'bench', '--out', str(out), '--img-size', '64', '--bench-quick', 'true',
            '--models', 'unet', 'deeplabv3plus_r50']  # fmt: skip
    assert main(args) == 0
    measured = json.loads(bench.read_text())
    assert [r['model'] for r in measured['results']] == ['unet', 'deeplabv3plus_r50']
    assert measured['env']['device'] != 'old session'
    stamp = bench.stat().st_mtime_ns
    assert main(args) == 0 and bench.stat().st_mtime_ns == stamp
