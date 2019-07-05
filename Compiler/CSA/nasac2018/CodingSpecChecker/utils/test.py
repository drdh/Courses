from sys import argv
from subprocess import call
from sys import stderr, stdout
import resource
from multiprocessing import Process
import os
import psutil

usage = """
Usage:
    xxx tool_path source_path build_path output_dir
"""

if len(argv) < 5:
    print(usage)
    exit()

tool_path, source_path, build_path, output_dir = argv[1:5]

modules = [
    "naming-check", "error-handling-check", "full-comment-check", "header-check", "init-in-need-check", "module-check"
]

def call_with_statistic(cmd, fresult, fstatistic=stdout):
    def fun():
        print("command: {}".format(' '.join(cmd)), file=fstatistic)
        call(cmd, stderr=fresult)
        # Get resource after running
        info = resource.getrusage(resource.RUSAGE_CHILDREN)
        time, mem = info.ru_utime, info.ru_maxrss
        print("time: {}s\nmem: {}MB".format(time, mem/1024), file=fstatistic)
        print("", file=fstatistic)
    p = Process(target=fun)
    p.start()
    p.join()

# Run without checker
cmd = [tool_path]
cmd += [('-no-' + m) for m in modules]
cmd += ['-b=' + build_path, source_path]
print("Running without checker")
with open(os.path.join(output_dir, 'no_module.txt'), 'w+') as fout:
    call_with_statistic(cmd, fresult=fout)

# Run with each single checker
for i in range(len(modules)):
    cmd = [tool_path] + argv[5:]
    cmd += [('-no-'+ m) for m in (modules[:i] + modules[i+1:])]
    cmd += ['-b=' + build_path, source_path]
    print("Running with " + modules[i])
    with open(os.path.join(output_dir, modules[i]+'.txt'), 'w+') as fout:
        call_with_statistic(cmd, fresult=fout)

# Run with all checkers
print("Running with all checkers")
with open(os.path.join(output_dir, 'all.txt'), 'w+') as fout:
    call_with_statistic([tool_path] + argv[5:] + ['-b='+build_path, source_path], fresult=fout)
