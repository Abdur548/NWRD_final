"""The public NWRD release: finding it, checking it against its manifest, the split, and reading
photographs, masks and cut windows.

The release (Anwar and Shafait, Zenodo doi:10.5281/zenodo.21672928, CC BY 4.0) holds 100
photographs of 10 sizes with pixel masks: <root>/{train,test}/{images,masks}/<id>.{jpg,png}.
docs/dataset.md describes it, and docs/protocol.md §2 explains the split.
"""

import glob
import hashlib
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics
ZENODO_URL = 'https://zenodo.org/api/records/21672928/files/NWRD.zip/content'
ZENODO_MD5 = '7b9193efebe1ed8c60b10c9f05d11e14'
ZENODO_BYTES = 608_898_312

# The authors' validation photographs: their loader shuffles the sorted training file names with
# NumPy seed 0 and keeps the first 10% (paper_validation_ids). Their test set is the test/ folder.
VAL_IDS = ('2', '30', '72', '87', '101', '118', '136', '149', '150')
# The release repeats ten scenes: nine pairs of files decode to identical pixels, and 149 is 99
# cropped and saved again; each copy has its own mask. These four training photographs are twins
# of validation or test photographs (value), so no model trains on them.
EXCLUDED_TWINS = {'79': '80', '86': '89', '96': '87', '99': '149'}
TWIN_PAIRS = (
    ('15', '32'),
    ('83', '128'),
    ('100', '137'),
    ('69', '141'),
    ('78', '81'),
    ('91', '98'),
    ('79', '80'),
    ('86', '89'),
    ('96', '87'),
    ('99', '149'),
)


def default_manifest_path() -> str:
    """data/manifest.json next to the package, in the repository and in a notebook's code/."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'manifest.json'
    )


def load_manifest(path: str | None = None) -> dict:
    """{'source': ..., 'files': {'train/images/10.jpg': {'md5': ..., 'bytes': ...}, ...}}"""
    with open(path or default_manifest_path(), encoding='utf-8') as f:
        return json.load(f)


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def photo_id(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def photo_folders(manifest: dict) -> dict[str, str]:
    """Photograph id -> the release folder holding it, 'train' or 'test'."""
    return {photo_id(n): n.split('/')[0] for n in manifest['files'] if '/images/' in n}


def paper_validation_ids(train_images: list[str]) -> list[str]:
    """The authors' validation draw (`BaseDataLoader._split_sampler`, validation_split 0.1) over
    the training images' paths sorted as text. The legacy global NumPy generator after
    np.random.seed(0) draws exactly what RandomState(0) does."""
    names = sorted(train_images)
    idx = np.arange(len(names))
    np.random.RandomState(0).shuffle(idx)
    return [photo_id(names[i]) for i in idx[: int(len(names) * 0.1)]]


def splits(manifest: dict) -> dict[str, list[str]]:
    """Photograph ids per split, each sorted by number: the authors' validation and test
    photographs, and their training photographs without EXCLUDED_TWINS."""
    folders = photo_folders(manifest)
    stray = [i for i in VAL_IDS if folders.get(i) != 'train']
    if stray:
        raise ValueError(f'validation photographs {stray} are not in the release train/ folder')
    in_train = [i for i, f in folders.items() if f == 'train']
    return {
        'train': sorted(
            (i for i in in_train if i not in VAL_IDS and i not in EXCLUDED_TWINS), key=int
        ),
        'val': sorted(VAL_IDS, key=int),
        'test': sorted((i for i, f in folders.items() if f == 'test'), key=int),
    }


def find_release(hint: str | None, manifest: dict) -> str:
    """The folder holding the release's train/ and test/: `hint` itself, or the first folder
    below it, then below Kaggle's /kaggle/input, where the manifest's first file sits."""
    probe = sorted(manifest['files'])[0].split('/')
    candidates = [hint] if hint else []
    candidates.append('/kaggle/input')
    for base in candidates:
        if os.path.isfile(os.path.join(base, *probe)):
            return base
        hits = sorted(glob.glob(os.path.join(glob.escape(base), '**', *probe), recursive=True))
        if hits:
            root = hits[0]
            for _ in probe:
                root = os.path.dirname(root)
            return root
    raise FileNotFoundError(
        f'no copy of the NWRD release under {candidates}; attach the Kaggle dataset '
        'bcap52/nwrd-wheat or pass --data-root (scripts/fetch_data.py downloads it)'
    )


def verify_release(root: str, manifest: dict) -> None:
    """Raises RuntimeError unless every file of the manifest is under `root`, byte for byte (its
    size and MD5). Files the manifest does not list are ignored."""
    missing, changed = [], []
    for name, entry in sorted(manifest['files'].items()):
        path = os.path.join(root, *name.split('/'))
        if not os.path.isfile(path):
            missing.append(name)
        elif os.path.getsize(path) != entry['bytes'] or md5_of(path) != entry['md5']:
            changed.append(name)
    if missing or changed:
        raise RuntimeError(
            f'the release under {root} does not match the manifest: {len(missing)} missing '
            f'{missing[:5]}, {len(changed)} changed {changed[:5]}'
        )


@dataclass
class Release:
    root: str
    folders: dict[str, str]  # photograph id -> 'train' or 'test', its folder in the release
    splits: dict[str, list[str]]  # 'train', 'val', 'test' -> photograph ids


def open_release(
    hint: str | None = None, manifest_path: str | None = None, limit: int = 0, log=print
) -> Release:
    """Finds the release, checks every file against the manifest, and splits it. With `limit`,
    each split keeps only its first `limit` photographs (smoke tests)."""
    manifest = load_manifest(manifest_path)
    root = find_release(hint, manifest)
    verify_release(root, manifest)
    log(f'data: {root}: all {len(manifest["files"])} files match the manifest')
    s = splits(manifest)
    if limit:
        s = {k: v[:limit] for k, v in s.items()}
    return Release(root, photo_folders(manifest), s)


def _file(root: str, folder: str, kind: str, pid: str) -> str:
    return os.path.join(root, folder, kind, f'{pid}.{"jpg" if kind == "images" else "png"}')


def read_mask(root: str, folder: str, pid: str) -> np.ndarray:
    """Rust (bool). The release's masks hold only the values 0 and 255."""
    path = _file(root, folder, 'masks', pid)
    grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if grey is None:
        raise FileNotFoundError(path)
    return grey > 127


def read_photo(root: str, folder: str, pid: str) -> tuple[np.ndarray, np.ndarray]:
    """One photograph (RGB uint8) and its rust mask (bool), of the same shape.

    The stored pixels are used as they are: every photograph has EXIF orientation 1 or none, and
    every mask matches its photograph's stored size, except photo 101's, which is 19 px narrower
    and 21 px shorter and lines up with its top-left corner (docs/dataset.md). A photograph is
    therefore cropped to a smaller mask at the top left; a larger mask is an error."""
    path = _file(root, folder, 'images', pid)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise FileNotFoundError(path)
    rust = read_mask(root, folder, pid)
    (h, w), (mh, mw) = bgr.shape[:2], rust.shape
    if mh > h or mw > w:
        raise ValueError(f'photo {pid}: mask {mw}x{mh} is larger than the photograph {w}x{h}')
    return cv2.cvtColor(bgr[:mh, :mw], cv2.COLOR_BGR2RGB), rust


def camera(root: str, folder: str, pid: str) -> str:
    """The camera model in the photograph's EXIF, or 'no EXIF'."""
    from PIL import Image

    with Image.open(_file(root, folder, 'images', pid)) as img:
        model = img.getexif().get(272)  # Model
    model = model.strip('\x00 ') if isinstance(model, str) else ''
    return model or 'no EXIF'


def train_transform():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ElasticTransform(p=0.5),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


class RustDataset(torch.utils.data.Dataset):
    """(image tensor, mask tensor of shape (1, H, W), index) for each cut window."""

    def __init__(self, img_dir: str, mask_dir: str, files, transform):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.files, self.transform = list(files), transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, name)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE) > 127
        out = self.transform(image=img, mask=mask.astype(np.float32))
        return out['image'], out['mask'].unsqueeze(0), idx
