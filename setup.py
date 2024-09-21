
import os
import re
import subprocess

from setuptools import setup, find_packages
from setuptools.command.install import install

class DependenciesCheckInstallCommand(install):
    """Check for graphviz."""
    def run(self):
        try:
            # Check if 'dot' (Graphviz executable) is available
            subprocess.check_call(['dot', '-V'],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Graphviz not installed or not found
            print("Graphviz is not installed. Please install it " +
                  "manually using 'sudo apt install graphviz' " +
                  "for Debian-based systems, or " +
                  "visit https://graphviz.org/download/ " +
                  "for other platforms.")
            # prompt the user to install it
            if input("Do you want to continue installation " +
                     "without Graphviz? [y/N]: ").lower() != 'y':
                print("Installation aborted due to missing Graphviz.")
                return

        install.run(self)

# Read the package __version__.py file
def get_version():
    version_file = os.path.join(os.path.dirname(__file__),
                                "retrain_pipelines", "__version__.py")
    with open(version_file) as f:
        version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                            f.read(), re.M)
        if version:
            return version.group(1)
        raise RuntimeError("Unable to find version string.")

setup(
    name='ml_pipelines',
    version=get_version(),
    packages=find_packages(),
    author="Aurelien-Morgan",
    license="Apache-2.0",
    install_requires=[
        'ipykernel>=6.20',
        'pandas>=2',
        'matplotlib>=3.4',
        'metaflow>=2.9',
        'wandb>=0.15',
        'plotly>=4',
        'jinja2>=3',
        'scikit-learn>=1',
        'numpy>=1.17.3,<2',
    ],
    cmdclass={
        'install': DependenciesCheckInstallCommand,
    },
)

