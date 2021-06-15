"""
Configuration file for various runtime environments for BEEP-EP.


"""



json_log_fmt = (
    '{"time": "%(asctime)s", "level": "%(levelname)s", '
    '"service": "%(service)s", "process": "%(process)d", '
    '"module": "%(module)s", "func": "%(funcName)s", '
    '"msg": "%(message)s"}'
)

CONFIG = {
    "local": {
        "logging": {
            "container": "Testing",
            "streams": ["stdout"],
            "fmt": None
        },
        "kinesis": {"stream": "local/beep/eventstream"},
    },
    "dev": {
        "logging": {
            "container": "Testing",
            "streams": ["stdout"],
            "fmt": None
        },
        "kinesis": {"stream": "local/beep/eventstream"},
    },
    "test": {
        "logging": {
            "container": "Testing",
            "streams": ["file"],
            "fmt": json_log_fmt
        },
        "kinesis": {"stream": "local/beep/eventstream"},
    },
    "stage": {
        "logging": {
            "container": "BEEP_EP",
            "streams": ["CloudWatch", "stdout"],
            "fmt": json_log_fmt
        },
        "kinesis": {"stream": "stage/beep/eventstream/stage"},
    },
    "prod": {},
}
