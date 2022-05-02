import os
from setuptools import setup, find_packages
from pip._internal.req import parse_requirements
from pip._internal.network.session import PipSession

this_dir = os.path.dirname(os.path.abspath(__file__))
pip_requirements = parse_requirements(
    os.path.join(this_dir, "requirements.txt"), PipSession())
pip_requirements_test = parse_requirements(
    os.path.join(this_dir, "requirements-test.txt"), PipSession())

reqs = [pii.requirement for pii in pip_requirements]
reqs_test = [pii.requirement for pii in pip_requirements_test]

readme_path = os.path.join(this_dir, "README.md")

with open(readme_path, "r") as f:
    long_description = f.read()

# description must be one line
description = "beep is a python package supporting Battery Estimation and Early Prediction of battery cycle life."

setup(name="beep",
      url="https://github.com/TRI-AMDD/beep",
      version="2022.5.2.14",
      description=description,
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=reqs,
      extras_require={"tests": reqs_test},
      entry_points='''
        [console_scripts]
        beep=beep.cmd:cli
        ''',
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      package_data={
          "beep.conversion_schemas": ["*.yaml", "*.md"],
          "beep.protocol.biologic_templates": ["*.mps", "*.csv", "*.json"],
          "beep.protocol.procedure_templates": ["*.000", "*.csv", "*.json",
                                                "*.yaml"],
          "beep.protocol.protocol_schemas": ["*.yaml", "*.txt"],
          "beep.protocol.schedule_templates": ["*.sdu", "*.csv", "*.json"],
          "beep.protocol-parameters": ["*.csv"],
          "beep.validation_schemas": ["*.yaml"],
          # "beep.model": ["*.model"],
          # "beep.features": ["*.yaml"]

      },
      include_package_data=True,
      author="AMDD - Toyota Research Institute",
      author_email="patrick.herring@tri.global",
      maintainer="Patrick Herring",
      maintainer_email="patrick.herring@tri.global",
      license="Apache",
      keywords=[
          "materials", "battery", "chemistry", "science",
          "electrochemistry", "energy", "AI", "artificial intelligence"
      ],
      )
