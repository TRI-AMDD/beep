"""
Configuration file for various runtime environments for BEEP-EP.


"""

config = {
    "local": {
        "logging": {"container": "Testing", "streams": ["file"]},
        "kinesis": {"stream": "local/beep/eventstream"},
    },
    "dev": {
        "logging": {"container": "Testing", "streams": ["file"]},
        "kinesis": {"stream": "local/beep/eventstream"},
    },
    "test": {
        "logging": {"container": "Testing", "streams": ["file"]},
        "kinesis": {"stream": "local/beep/eventstream"},
    },
    "stage": {
        "logging": {"container": "BEEP_EP", "streams": ["CloudWatch", "stdout"]},
        "kinesis": {"stream": "stage/beep/eventstream/stage"},
    },
    "prod": {},
}
