"""Setup module for app-lal-gwave package
"""

import pathlib

from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(name="gwsciapp",
      version="0.1.0",
      description="GWSci Apps",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/JWKennington/app-lal-gwave",
      author="J. W. Kennington",
      author_email="jameswkennington@gmail.com",
      classifiers=[
          "Development Status :: 3 - Alpha",
          # Pick your license as you wish
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Programming Language :: Python :: 3 :: Only",
      ],
      keywords="topic reference, bibliography, education, reference index",
      packages=find_packages(),
      python_requires=">=3.7, <4",
      install_requires=["pyyaml"],
      extras_require={  # Optional
          "dev": ["check-manifest"],
          "test": ["pytest", "pytest-cov"],
      },
      # entry_points={  # Optional
      #     "console_scripts": [
      #         "torus=scripts.torus:main",
      #     ],
      # },
      project_urls={  # Optional
          "Bug Reports": "https://github.com/JWKennington/app-lal-gwave/issues",
          "Source": "https://github.com/JWKennington/app-lal-gwave",
      },
      )
