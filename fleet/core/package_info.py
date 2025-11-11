MAJOR = 0
MINOR = 0
PATCH = 0
PRE_RELEASE = 'rc0'

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])
__package_name__ = 'fleet_core'
__contact_names__ = 'PadldePaddle'
__contact_emails__ = 'Paddle-better@baidu.com'
__homepage__ = 'https://www.paddlepaddle.org.cn/documentation/guides/index_cn.html'
__repository_url__ = 'https://github.com/PaddlePaddle/PaddleFleet'
__download_url__ = 'https://github.com/PaddlePaddle/PaddleFleet/releases'
__description__ = (
    'PaddleFleet - Core Functional Library for Large Scale Distributed Training'
)
__license__ = 'Apache Software License',
__keywords__ = (
    'paddlepaddle,deep learning, machine learning, gpu, NLP, language, transformer, paddle'
)
