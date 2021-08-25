"""Setup modelenv."""
from setuptools import setup, find_packages
from pathlib import Path
import typing

from modelenv import __version__


readme_path = Path(__file__).absolute().parent.joinpath('README.md')
long_description = readme_path.read_text(encoding='utf-8')


def install_requires() -> typing.List[str]:
    """Find the install requires strings from requirements.txt"""
    requires = []
    with open(
        Path(__file__).absolute().parent.joinpath('requirements.txt'), "r"
    ) as requirments_txt_handle:
        requires = [
            x
            for x in requirments_txt_handle
            if not x.startswith(".") and not x.startswith("-e")
        ]
    return requires


setup(
    name='modelenv',
    version=__version__,
    description='An OpenAI gym environment for searching for optimal machine learning models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: GPL 2.0 License',
        'Programming Language :: Python :: 3',
        'Framework :: OpenAI Gym',
    ],
    keywords='openai openai-gym-environments openai-gym gym nas hyperparameter-search torch tensorflow scipy',
    url='https://github.com/8W9aG/modelenv',
    author='Will Sackfield',
    author_email='will.sackfield@gmail.com',
    license='GPL2.0',
    install_requires=install_requires(),
    zip_safe=False,
    packages=find_packages()
)
