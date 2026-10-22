"""Downloads the public NWRD release from Zenodo, checks it, and extracts it safely.

    python scripts/fetch_data.py                         # download to data/raw/, check, extract
    python scripts/fetch_data.py --zip path/to/NWRD.zip  # an archive downloaded before

Nothing is extracted unless the archive matches Zenodo's MD5 and every entry is one of the
expected .jpg or .png files under train/ or test/ (no absolute paths, no '..', no oversized
entries). The extracted files are then checked against data/manifest.json. `--write-manifest`
(maintainers) writes that manifest instead, from a fresh extraction of an archive whose MD5
matched.

Do not get the release from the dataset's GitHub repository: it was compromised in October 2026
and opening a clone in VS Code runs a credential stealer (docs/dataset.md §1).
"""

import argparse
import json
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # run from a checkout without installing the package

from nwrd_public.dataset import (  # noqa: E402
    ZENODO_BYTES,
    ZENODO_MD5,
    ZENODO_URL,
    default_manifest_path,
    load_manifest,
    md5_of,
    verify_release,
)

LAYOUT = {'train/images': 90, 'train/masks': 90, 'test/images': 10, 'test/masks': 10}
MAX_ENTRY_BYTES = 50_000_000  # the largest file is 7.8 MB (train/images/104.jpg)
SOURCE = (
    'NWRD.zip from Zenodo (Anwar and Shafait, doi:10.5281/zenodo.21672928, v1.0, CC BY 4.0), '
    f'{ZENODO_BYTES:,} bytes, MD5 {ZENODO_MD5}'
)


def download(url: str, path: str) -> None:
    """Streams `url` to `path`, resuming a partial file with an HTTP Range request."""
    have = os.path.getsize(path) if os.path.exists(path) else 0
    req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # the server ignored the range: start again
            have = 0
        total, done = have + int(r.headers.get('Content-Length', 0)), have
        with open(path, 'ab' if have else 'wb') as f:
            for block in iter(lambda: r.read(1 << 20), b''):
                f.write(block)
                done += len(block)
                print(f'\r{done / 1e6:.0f} / {total / 1e6:.0f} MB', end='', flush=True)
    print()


def check_archive(path: str) -> None:
    digest = md5_of(path)
    if digest != ZENODO_MD5:
        raise SystemExit(f'{path}: MD5 {digest}, but Zenodo publishes {ZENODO_MD5}: not extracting')


def safe_extract(zip_path: str, dst: str, layout: dict[str, int] = LAYOUT) -> dict[str, int]:
    """Extracts the archive into `dst` once every entry has passed the checks; returns the number
    of files per folder, which must equal `layout`."""
    dst = os.path.abspath(dst)
    with zipfile.ZipFile(zip_path) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        counts = {}
        for i in infos:
            name = i.filename.replace('\\', '/')
            parts = name.split('/')
            target = os.path.abspath(os.path.join(dst, *parts))
            if (
                name.startswith('/')
                or '..' in parts
                or ':' in name
                or not target.startswith(dst + os.sep)
                or not name.lower().endswith(('.jpg', '.png'))
            ):
                raise SystemExit(f'unexpected entry {i.filename!r}: not extracting')
            if i.file_size > MAX_ENTRY_BYTES or i.file_size > 200 * max(i.compress_size, 1):
                raise SystemExit(f'suspicious size for {i.filename!r}: not extracting')
            counts[os.path.dirname(name)] = counts.get(os.path.dirname(name), 0) + 1
        if counts != layout:
            raise SystemExit(f'unexpected layout {counts}, expected {layout}: not extracting')
        z.extractall(dst, members=infos)
    return counts


def write_manifest(root: str, path: str, layout: dict[str, int] = LAYOUT) -> None:
    """MD5 and size of every photograph (.jpg) and mask (.png) under `root`, as
    data/manifest.json. The number of each must equal `layout`; anything else in the folders
    (a Thumbs.db, say) is left out."""
    files = {}
    for folder, expected in layout.items():
        ext = '.jpg' if folder.endswith('images') else '.png'
        names = sorted(n for n in os.listdir(os.path.join(root, folder)) if n.endswith(ext))
        if len(names) != expected:
            raise SystemExit(f'{folder} holds {len(names)} {ext} files, expected {expected}')
        for name in names:
            p = os.path.join(root, folder, name)
            files[f'{folder}/{name}'] = {'md5': md5_of(p), 'bytes': os.path.getsize(p)}
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump({'source': SOURCE, 'files': files}, f, indent=1, sort_keys=True)
        f.write('\n')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--zip', help='an archive downloaded before (default: <dest>/NWRD.zip)')
    ap.add_argument('--dest', default=os.path.join(ROOT, 'data', 'raw'))
    ap.add_argument('--manifest', default=default_manifest_path())
    ap.add_argument(
        '--write-manifest', action='store_true', help='write the manifest from this archive'
    )
    a = ap.parse_args(argv)

    os.makedirs(a.dest, exist_ok=True)
    zip_path = a.zip or os.path.join(a.dest, 'NWRD.zip')
    if not a.zip and not (os.path.exists(zip_path) and os.path.getsize(zip_path) == ZENODO_BYTES):
        print('downloading', ZENODO_URL)
        download(ZENODO_URL, zip_path)
    check_archive(zip_path)
    print('archive MD5 matches Zenodo:', ZENODO_MD5)
    out = os.path.join(a.dest, 'NWRD')
    if os.path.isdir(out) and os.listdir(out):
        if a.write_manifest:  # the manifest must describe the checked archive, nothing else
            raise SystemExit(f'--write-manifest needs a fresh extraction: remove {out} first')
        print(f'{out} exists: checking it instead of extracting again')
    else:
        print('extracted', safe_extract(zip_path, out), 'to', out)
    if a.write_manifest:
        write_manifest(out, a.manifest)
        print('wrote', a.manifest)
    manifest = load_manifest(a.manifest)
    verify_release(out, manifest)
    print(f'{out}: all {len(manifest["files"])} files match {a.manifest}')


if __name__ == '__main__':
    main()
