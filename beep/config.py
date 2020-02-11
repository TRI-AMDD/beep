"""
Configuration file for various runtime environments for BEEP-EP.


"""

config = {

    'local': {
        'logging': {
            'container': 'Testing',
            'streams': ['CloudWatch']
        }
    },

    'dev': {
        'logging': {
            'container': 'Testing',
            'streams': ['file']
        }
    },

    'test': {
        'logging': {
            'container': 'Testing',
            'streams': ['CloudWatch']
        }
    },

    'stage': {
        'logging': {
            'container': 'BEEP_EP',
            'streams': ['CloudWatch']
        }
    },

    'prod': {}

}
