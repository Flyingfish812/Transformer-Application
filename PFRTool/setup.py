from setuptools import setup, find_packages

setup(
    name='PFRTool',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'torch',
        'matplotlib',
    ],
    author='Flyingfish812',
    author_email='flyingfish812@foxmail.com',
    description='A library for Physic Field Reconstruction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/flyingfish812/Transformer-Application',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)