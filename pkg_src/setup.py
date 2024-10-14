
import os
import re
import shutil
import subprocess

LICENSE = "LICENSE"
if os.path.exists(os.path.join("..", LICENSE)):
    shutil.copy(os.path.join("..", LICENSE), LICENSE)
# to avoid having to maintain 1 for Gihub and 1 for PyPi
README = "README.md"
if os.path.exists(os.path.join("..", README)):
    shutil.copy(os.path.join("..", README), README)

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


setup(
    name='retrain_pipelines',
    author="Aurelien-Morgan",
    license="Apache-2.0",
    cmdclass={
        'install': DependenciesCheckInstallCommand,
    },
)

