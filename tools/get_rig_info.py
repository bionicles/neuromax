from subprocess import check_output
from collections import Iterable
from pprint import pprint
import itertools
import platform
import inspect

GPU_QUERIES = [
    'gpu_name',
    'driver_version',
    # 'vbios_version',
    # 'pstate',
    'memory.total',
    'temperature.gpu',
    # 'temperature.memory',
    # 'power.draw',
    'power.limit',
    'clocks.gr',
    'clocks.sm',
    'clocks.mem',
    'clocks.video'
]

COMMANDS = {
    q: f'nvidia-smi --query-gpu={q} --format=csv,noheader'
    for q in GPU_QUERIES}

BLACKLIST = [
    'linux_distribution',
    'python_version_tuple',
    'java_ver',
    'uname',
    'version',
    'dist'
]


# https://stackoverflow.com/questions/9727688/how-to-get-the-cuda-version
def get_cuda_version():
    return open('/usr/local/cuda/version.txt').readlines()[0][13:-2]


def get_numbers(S):
    return [int(s) for s in S.split() if s.isdigit()]


# https://stackoverflow.com/questions/31326015/how-to-verify-cudnn-installation
def get_cudnn_version():
    # cudnn_h_path = '/usr/local/cuda/include/cudnn.h'
    cudnn_h_path = '/usr/include/cudnn.h'
    N = get_numbers(
            check_output(
                f'cat {cudnn_h_path} | grep CUDNN_MAJOR -A 2', shell=True))
    V = f'{N[0]}.{N[1]}.{N[2]}'
    return V


def get_ubuntu_version():
    return ' '.join(
        check_output('lsb_release -sdc', shell=True).decode().split())


def get_memory_info():
    S = check_output('head -1 /proc/meminfo', shell=True).decode()
    return f'{get_numbers(S)[0] / 1e6} GB'


def get_cpu_info():
    cpu_info = {}
    for key in ['model name', 'cpu cores', 'cache size']:
        cpu_info[key] = decode(check_output(
            f"cat /proc/cpuinfo | grep '{key}' | head -n 1 | cut -d' ' -f 3-",
            shell=True))
    return cpu_info


def no_underscore(string):
    try:
        return '_' not in list(string[:2])
    except Exception as e:
        print(e, string)
        return '_' not in list(string)


def _flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def flatten(l):
    if isinstance(l, Iterable) and not isinstance(l, str):
        return " ".join(list(filter(None, _flatten(l))))
    else:
        return l


def drop_falsy(l):
    return list(itertools.filterfalse(True, flatten(l)))


def empty(x):
    if isinstance(x, str) and x:
        return False
    elif isinstance(x, Iterable):
        return len(drop_falsy(x))
    else:
        return False


def decode(bytes):
    return bytes.decode().rstrip("\n")


def invoke(key, module_or_command):
    if inspect.ismodule(module_or_command):
        module_or_command = getattr(module_or_command, key)
    if isinstance(module_or_command, str):
        if no_underscore(module_or_command):
            output = check_output(module_or_command, shell=True)
            if isinstance(output, bytes):
                output = decode(output)
            # print("invoke output", output)
            return output
    elif callable(module_or_command):
        return module_or_command()
    # print(f"INVOKE FAILED ON {key}")
    return None


def try_to_set(key, info, module_or_command):
    if key not in BLACKLIST and no_underscore(key):
        try:
            value = invoke(key, module_or_command)
            # print('invoke output type', type(value))
            value = flatten(value)
            if value:
                info[key] = value
        except Exception as e:
            print('exception in try_to_set', key, module_or_command, e)
    return info


def remove_redundant_values(d):
    d2 = {}
    for key, value in d.items():
        if value not in d2.values():
            d2[key] = value
    return d2


def delete_stuff_in_platform_string(d):
    d2 = {k: v for k, v in d.items() if v not in d['platform']}
    d2['platform'] = d['platform']
    return d2


def get_rig_info():
    platform_info = {}
    for key in dir(platform):
        platform_info = try_to_set(key, platform_info, platform)
    platform_info = remove_redundant_values(platform_info)
    platform_info = delete_stuff_in_platform_string(platform_info)
    platform_info['os'] = get_ubuntu_version(),
    gpu_info = {'cuda': get_cuda_version(), 'cudnn': get_cudnn_version()}
    for key, command in COMMANDS.items():
        gpu_info = try_to_set(key, gpu_info, command)
    info = {
        'ram': get_memory_info(),
        'cpu': get_cpu_info(),
        'platform': platform_info,
        'gpu': remove_redundant_values(gpu_info)}
    pprint(info)
    return info


get_rig_info()
# get_memory_info()
# get_cpu_info()
# get_ubuntu_version()
# get_cudnn_version()
