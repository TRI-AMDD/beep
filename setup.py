from setuptools import setup, find_packages

setup(name="beep",
      url="https://github.com/ToyotaResearchInstitute/beep",
      version="2020.2.22",
      packages=find_packages(),
      install_requires=["numpy==1.18.1",
                        "monty==3.0.2",
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
                        "ruamel.yaml==0.16.5",
                        "xmltodict==0.12.0",
                        "tables==3.6.1"
                        ],
      extras_require={
          "tests": ["nose",
                    "coverage",
                    "pylint",
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