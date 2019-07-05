import subprocess
import sys
from PIL import Image
from Hook import HookGenerator

# args
origin_lib_full = '../lib/libeculid.so'
origin_lib_path = '../lib'
origin_lib_name = 'eculid'

activate_path = '../venv/bin/activate'
python_exec_path = 'python3'
target = '../src/test.py'

wrapper_lib_path = '../lib'
wrapper_lib_name = 'wrapper'

loader_path = '../bin/tryload'

print('generator symbols ...\n')
gen = HookGenerator(origin_lib_full)
gen.fetch()
gen.gen()

process = subprocess.Popen("/bin/bash", shell=True, stdin=subprocess.PIPE, stdout=sys.stdout,
                           stderr=sys.stderr)

print('compile hooker...\n')
# there are some bugs in direct output to g++

# command = 'echo "{:s}" | g++ -Wl,--no-as-needed ' \
#           '-L {:s} -l{:s} -L {:s} -l{:s} -fPIC -shared -o libhook.so -std=c++14 -xc++ '.format(
#     gen.src,
#     origin_lib_path,
#     origin_lib_name,
#     wrapper_lib_path,
#     wrapper_lib_name
# )
#
# process.stdin.write((command + '\n').encode('utf-8'))

with open('hook.cpp', 'w') as f:
    f.write(gen.src)

command = 'cmake CMakeLists.txt'
print(command)
process.stdin.write((command + '\n').encode('utf-8'))

command = 'make'
print(command)
process.stdin.write((command + '\n').encode('utf-8'))

print('enter virtual environment...\n')
command = 'source {:s}'.format(activate_path)
print(command)
process.stdin.write((command + '\n').encode('utf-8'))

print('run program...\n')

command = 'export LD_LIBRARY_PATH="{:s}"'.format(origin_lib_path)
print(command)
process.stdin.write((command + '\n').encode('utf-8'))

command = 'LD_PRELOAD=../lib/libhook.so {:s} -m memory_profiler --callanalysis {:s}'.format(python_exec_path, target)
print(command)
process.stdin.write((command + '\n').encode('utf-8'))

command = loader_path
print(command)
process.stdin.write((command + '\n').encode('utf-8'))

process.communicate()

img=Image.open('graph.png')
img.show()
