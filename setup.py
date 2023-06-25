"""
setup file for the tipeval package
"""

import os
import re
from setuptools import setup, find_packages
from typing import List


def get_version() -> str:
    """Obtain the version information from the __init__.py file"""

    file = os.path.join('tipeval', '__init__.py')
    version_string = re.compile(r"""__version__[\s]*=[\s]*["|'](.*)["|']""")
    with open(file, 'r') as f:
        version = str(re.search(version_string, f.read()))
        print(version)
    return version


setup(
    name='tipeval',
    version=get_version(),
    packages=find_packages(),
    include_package_data=True,
    package_data={'tipeval': ["config.yaml",
                              "core/configuration_intern.yaml",
                              "ui/resources/icons/*",
                              "ui/resources/ui_files/*",
                              "ui/resources/*"]},
    entry_points={
        'console_scripts': [
            'tipeval-gui=tipeval.ui.main_user_interface:main'
        ]
    },
    author='Christian Saringer',
    author_email='christian.saringer@unileoben.ac.at; christian_saringer@gmx.net',
)
