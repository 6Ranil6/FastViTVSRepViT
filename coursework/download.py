import hashlib
import tarfile
import urllib.request
from .core import DATA, ROOT, sha256, write_json

FILES = {
    '102flowers.tgz': '52808999861908f626f3c1f4e79d11fa',
    'imagelabels.mat': 'e0620be6f572b9609742df49c70aed4d',
    'setid.mat': 'a5357ecc9cb78c4bef273ce3793fc85c',
}

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, expected in FILES.items():
        path = DATA / name
        def valid():
            return path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == expected
        url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/' + name
        if not valid():
            print('Downloading', url, flush=True)
            tmp = path.with_suffix(path.suffix + '.part')
            urllib.request.urlretrieve(url, tmp)
            if hashlib.md5(tmp.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Checksum mismatch: {tmp}')
            tmp.replace(path)
        manifest[name] = {'source': url, 'md5': expected, 'sha256': sha256(path)}
    with tarfile.open(DATA / '102flowers.tgz') as archive:
        archive.extractall(DATA, filter='data')
    from scipy.io import loadmat
    splits = loadmat(DATA / 'setid.mat')
    sets = [set(splits[k].reshape(-1).tolist()) for k in ('trnid', 'valid', 'tstid')]
    assert [len(x) for x in sets] == [1020, 1020, 6149]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(list((DATA / 'jpg').glob('*.jpg'))) == 8189
    write_json(DATA / 'manifest.json', manifest)
    write_json(ROOT / 'data/labels.json', [f'Flower class {i+1:03d}' for i in range(102)])
    print('Verified: 8,189 images; train=1,020, val=1,020, test=6,149.', flush=True)

if __name__ == '__main__':
    main()
