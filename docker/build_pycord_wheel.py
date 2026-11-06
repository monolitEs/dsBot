"""Build the pinned Pycord wheel from its verified upstream source archive."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

REVISION = '10a5e8cf13bd8db21f95baf62cf866056d1e1091'
SOURCE_SHA256 = 'f65c2b25d004fb847497334d0e7bbede3158c2cda9f3769e922e8e2bc6abbde8'
VERSION = '2.8.1.dev0+g10a5e8c'
WHEEL = f'py_cord-{VERSION}-py3-none-any.whl'


def build(archive: Path, output: Path):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Pycord source SHA256 mismatch')
    with tempfile.TemporaryDirectory(prefix='pycord-build-') as temporary:
        work = Path(temporary)
        source = work / 'source'
        source.mkdir()
        with tarfile.open(archive) as upstream:
            upstream.extractall(source, filter='data')
        sdk = source / f'pycord-{REVISION}'
        project = sdk / 'pyproject.toml'
        before = project.read_text(encoding='utf-8')
        marker = '    "discord.voice",'
        if before.count(marker) != 1:
            raise ValueError('Unexpected Pycord package list')
        after = before.replace(marker, marker + '\n    "discord.voice.receive",'
                               '\n    "discord.voice.packets",\n    "discord.voice.utils",')
        project.write_text(after, encoding='utf-8', newline='\n')
        environment = dict(os.environ, SETUPTOOLS_SCM_PRETEND_VERSION=VERSION,
                           SOURCE_DATE_EPOCH='315532800')
        wheels = work / 'wheels'
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps',
                        '--wheel-dir', str(wheels), str(sdk)], env=environment, check=True)
        wheel = wheels / WHEEL
        with zipfile.ZipFile(wheel) as package:
            files = [name for name in package.namelist()
                     if name.startswith('discord/') and name.endswith('.py')]
            expected = {p.relative_to(sdk).as_posix() for p in (sdk / 'discord').rglob('*.py')}
            if set(files) != expected or len(files) != 175:
                raise ValueError('Incomplete Pycord wheel')
            for name in files:
                if package.read(name) != (sdk / name).read_bytes():
                    raise ValueError(f'SDK source changed: {name}')
        output.mkdir(parents=True, exist_ok=True)
        destination = output / WHEEL
        if destination.exists() and destination.read_bytes() != wheel.read_bytes():
            raise ValueError('Existing wheel differs; inspect build tools before replacing')
        shutil.copyfile(wheel, destination)
        print(f'{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'vendor/pycord')
    args = parser.parse_args()
    build(args.archive, args.output)
