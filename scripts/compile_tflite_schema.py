import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
FLATBUFFERS_REPO: str = 'https://github.com/google/flatbuffers.git'
CMAKE_GENERATOR = 'Ninja'
OUTPUT_DIR = PROJECT_ROOT / 'src' / 'echonous' / 'exporters'


def build_flatbuffers_compiler() -> None:
    repo_dir = PROJECT_ROOT / '.cache' / 'repos' / 'flatbuffers'
    build_dir = PROJECT_ROOT / '.cache' / 'build' / 'flatbuffers'
    build_dir.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        subprocess.check_call(['git', 'clone', FLATBUFFERS_REPO, str(repo_dir)])
    else:
        subprocess.check_call(['git', 'pull'], cwd=repo_dir)
    subprocess.check_call([
        'cmake', '-G', CMAKE_GENERATOR,
        '-B', str(build_dir),
        '-S', str(repo_dir)
    ])
    subprocess.check_call(['cmake', '--build', str(build_dir)])

def compile_schema(schema: str | list[str]) -> None:
    flatc = PROJECT_ROOT / '.cache' / 'build' / 'flatbuffers' / 'flatc'

    schema_path = SCRIPT_DIR / schema
    output_path = OUTPUT_DIR
    print(schema_path)
    print(output_path)
    subprocess.check_call([
        str(flatc), '--python', '--gen-object-api', '-o', str(output_path), str(schema_path)
    ])

build_flatbuffers_compiler()
compile_schema('tflite.fbs')
compile_schema('tflite_metadata.fbs')