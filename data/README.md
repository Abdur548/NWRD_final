# Data

`manifest.json` (in git) lists the MD5 and size of each of the 200 files of the public NWRD
release, taken from Zenodo's `NWRD.zip` (doi:10.5281/zenodo.21672928, 608,898,312 bytes, MD5
`7b9193efebe1ed8c60b10c9f05d11e14`, CC BY 4.0). Every run checks the release against it before
training and stops on any difference; the notebooks embed a copy.

`raw/` (not in git) holds the release itself:

```bash
python scripts/fetch_data.py                        # download, check the MD5, extract, verify
python scripts/fetch_data.py --zip path/to/NWRD.zip # an archive downloaded before
```

which gives `raw/NWRD.zip` (when downloaded) and `raw/NWRD/{train,test}/{images,masks}/`. The
Kaggle copy `bcap52/nwrd-wheat` has the same 200 file names and byte sizes; the runs there check
each file's MD5 before training. Never take the release from the dataset's GitHub repository
(`docs/dataset.md` §1).
