"""
Configuration file for various runtime environments for BEEP-EP.


"""



json_log_fmt = {"fmt": (
    '{"time": "%(asctime)s", "level": "%(levelname)s", '
    '"service": "%(service)s", "process": "%(process)d", '
    '"module": "%(module)s", "func": "%(funcName)s", '
    '"msg": "%(message)s"}'
)
}

human_log_fmt = {
    "fmt": "%(asctime)s %(levelname)-8s %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S"
}


CONFIG = {
    "local": {
        "logging": {
            "container": "Testing",
            "streams": ["stdout"],
            "logger_args": human_log_fmt
        },
        "kinesis": {"stream": "local/beep/eventstream"},
    },
    "dev": {
        "logging": {
            "container": "Testing",
            "streams": ["stdout"],
            "logger_args": human_log_fmt
        },
        "kinesis": {"stream": "local/beep/eventstream"},
    },
    "test": {
        "logging": {
            "container": "Testing",
            "streams": ["file"],
            "logger_args": json_log_fmt
        },
        "kinesis": {"stream": "local/beep/eventstream"},
    },
    "stage": {
        "logging": {
            "container": "BEEP_EP",
            "streams": ["CloudWatch", "stdout"],
            "logger_args": json_log_fmt
        },
        "kinesis": {"stream": "stage/beep/eventstream/stage"},
    },
    "prod": {},
}
