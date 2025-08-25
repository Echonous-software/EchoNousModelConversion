"""Utilities for fetching the latest models from development repos"""
import datetime
import shutil
import subprocess
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXTRACT_ROOT = PROJECT_ROOT / 'src' / 'echonous' / '_vendor'


def main():
    repos_path = Path(__file__).parent / 'repos.yaml'
    with repos_path.open('r') as f:
        repos = yaml.safe_load(f)

    for repo_config in repos:
        process_repo_config(repo_config)


class Repo:
    """Container to run git commands scoped to a particular repo."""

    def __init__(self, name: str, path: Path, url: str, refspec: str):
        self.name = name
        self.path = path
        self.url = url
        self.refspec = refspec

    def git(self, *args: str) -> str:
        """Run git command and return output."""
        return subprocess.check_output(
            ['git'] + list(args),
            cwd=self.path,
            text=True
        ).strip()


def process_repo_config(repo_config: dict):
    """Sync a repo and run post-sync steps"""
    if len(repo_config) != 1:
        raise ValueError(f'Expected one repo per config, found {len(repo_config)}')

    name = list(repo_config.keys())[0]
    config = repo_config[name]

    print(f"Syncing {name} from {config['url']} @ {config['refspec']}")
    repo = sync_repo_to_refspec(name, config['url'], config['refspec'])

    for step in config.get('steps', []):
        if 'patch' in step:
            task_patch(repo, step['patch'])
        elif 'copy' in step:
            task_copy(repo, step['copy'])
        else:
            raise KeyError(f'Unknown step type: {list(step.keys())}')

    write_update_log(repo)

def get_cache_dir() -> Path:
    """Get cache directory for development."""
    cache_dir = PROJECT_ROOT / '.cache' / 'repos'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def sync_repo_to_refspec(name: str, url: str, refspec: str) -> Repo:
    """Synchronize an external git repository to a given refspec."""
    repo_path = get_cache_dir() / name
    repo = Repo(name, repo_path, url, refspec)

    if not repo_path.exists():
        print(f'Initializing repo {name}...')
        repo_path.mkdir(parents=True)
        repo.git('init')
        repo.git('remote', 'add', 'origin', url)
    elif repo_at_refspec(repo, refspec):
        return repo

    print(f'Fetching {refspec} in repo {name}...')
    repo.git('fetch', '--depth', '1', 'origin', refspec)
    repo.git('reset', '--hard', 'FETCH_HEAD')
    repo.git('clean', '-fd')

    return repo


def repo_at_refspec(repo: Repo, refspec: str) -> bool:
    current = repo.git('rev-parse', 'HEAD')
    target = repo.git('rev-parse', f'{refspec}^{{commit}}')
    return current == target

def task_patch(repo: Repo, patchfile: str):
    """Apply a patch file to the repo. Patchfile is relative to SCRIPT_DIR."""
    patch_path = SCRIPT_DIR / patchfile
    repo.git('restore', '.')
    repo.git('apply', str(patch_path.absolute()))


def task_copy(repo: Repo, args: dict):
    """Copy a list of files from the repo into src/echonous/_vendor/{repo.name}."""
    strip_directories = args.get('strip_directories', 0)
    dest_root = EXTRACT_ROOT / repo.name
    dest_root.mkdir(parents=True, exist_ok=True)

    for file in args['files']:
        source = repo.path / file
        relative_path = Path(*Path(file).parts[strip_directories:])
        dest = dest_root / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

def write_update_log(repo: Repo):
    log_path = EXTRACT_ROOT / repo.name / 'repo.yaml'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'url': repo.url,
        'refspec': repo.refspec,
        'commit_id': repo.git('rev-parse', 'HEAD'),
        'sync_time': datetime.datetime.now().isoformat()
    }
    with log_path.open('wt') as f:
        yaml.safe_dump(metadata, f)

if __name__ == '__main__':
    main()
