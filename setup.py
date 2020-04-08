from setuptools import setup, find_packages


# DESCRIPTION must be one line
DESCRIPTION = "beep is a python package supporting Battery Estimation and Early Prediction of battery cycle life."
LONG_DESCRIPTION = """
Beep is software designed to support Battery Estimation and Early Prediction
of cycle life corresponding to the research of the
[d3batt program](https://d3batt.mit.edu/) and the
[Toyota Research Institute](http://www.tri.global/accelerated-materials-design-and-discovery/).


Beep enables parsing and handing of electrochemical battery cycling data
via data objects reflecting cycling run data, experimental protocols,
featurization, and modeling of cycle life.  Currently beep supports
arbin and maccor cyclers.
"""

setup(name="beep",
      url="https://github.com/TRI-AMDD/beep",
      version="2020.4.5",
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=["numpy==1.18.1",
                        "monty[yaml]==3.0.2",
                        "scipy==1.4.1",
                        "scikit-learn==0.20.3",
                        "pandas==1.0.1",
                        "cerberus==1.3.2",
                        "tqdm==4.43.0",
                        "boto3==1.12.4",
                        "botocore==1.15.4",
                        "pytz==2019.3",
                        "watchtower==0.7.3",
                        "psycopg2==2.7.7",
                        "docopt==0.6.2",
                        "msgpack-python==0.5.6",
                        "python-dateutil==2.8.0",
                        "xmltodict==0.12.0",
                        "tables==3.6.1"
                        ],
      extras_require={
          "tests": ["pytest",
                    "pytest-cov",
                    "coveralls",
                    "memory_profiler",
                    "matplotlib"]
      },
      entry_points={
          "console_scripts": [
              "collate = beep.collate:main",
              "validate = beep.validate:main",
              "structure = beep.structure:main",
              "featurize = beep.featurize:main",
              "run_model = beep.run_model:main",
              "generate_protocol = beep.generate_protocol:main"
          ]
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      package_data={
          "beep.conversion_schemas": ["*.yaml", "*.md"],
          "beep.procedure_templates": ["*.000", "*.csv", "*.json"],
          "beep.validation_schemas": ["*.yaml"],
          "beep.model": ["*.model"],
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