import os.path
import shutil
import sys
import tempfile

TESTDIR = os.path.abspath(os.path.dirname(__file__))


def simms_executable() -> str:
    """
    Absolute path to the installed ``simms`` console script.

    Running the suite as ``python -m pytest`` does not put the interpreter's
    script directory on PATH, so look there first and only then fall back to
    PATH.
    """
    candidate = os.path.join(os.path.dirname(sys.executable), "simms")
    if os.path.exists(candidate):
        return candidate

    found = shutil.which("simms")
    if found:
        return found

    raise RuntimeError(
        f"The 'simms' console script was not found next to {sys.executable} or on PATH. Install the package first."
    )


class InitTest:
    def __init__(self):
        self.test_files = []

    def random_named_file(self, suffix: str = None):
        if not hasattr(self, "test_files"):
            self.test_files = []

        file_obj = tempfile.NamedTemporaryFile(suffix=suffix, dir=TESTDIR, delete=False)
        name = file_obj.name
        file_obj.close()

        self.test_files.append(name)
        return name

    def random_named_directory(self, suffix: str = None):
        if not hasattr(self, "test_files"):
            self.test_files = []

        name = tempfile.mkdtemp(suffix=suffix, dir=TESTDIR)

        self.test_files.append(name)
        return name

    def __del__(self):
        for path in getattr(self, "test_files", []):
            if os.path.exists(path):
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError as e:
                        print(f"Error deleting file '{path}': {e}")
                elif os.path.isdir(path):
                    try:
                        shutil.rmtree(path)
                    except OSError as e:
                        print(f"Error deleting directory '{path}': {e}")
