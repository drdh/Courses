import re
import os

template_include = '#ifndef _GNU_SOURCE\n#define _GNU_SOURCE\n#endif\n' \
                   '#include \"Global_Data.h\"\n#include \"Sync.h\"\n#include \"hook.h\"\n'
template_define_VG = "Global_data *VG = new Global_data();\nSync saver;\n"
template_namespace_entry = 'namespace {:s} {{'
template_namespace_exit = '}'
template_func_pointer = 'typedef {:s} (*PF{:d})({:s});'
template_func_entry = '{:s} {:s}({:s}) {{'

template_debug_load = 'VG = saver.Load();'
template_debug_entry = 'VG->enter_function("{:s}", "{:s}", 1);'
template_debug_exit = 'VG->exit_function();'
template_debug_save = 'saver.Save(VG);'

template_dlsym = 'PF{:d} ori = (PF{:d}) dlsym(RTLD_NEXT, "{:s}");'
template_dlsym_error = 'if (!ori) {{\n{st}\tfprintf(stderr, "%s\\n", dlerror());\n{st}\texit(1);\n{st}}}'
template_exec_no_ret = 'ori({:s});'
template_exec_with_ret = '{:s} result = ori({:s});'
template_exec_ret = 'return result;'
template_func_exit = '}'



class HookGenerator():
    def __init__(self, filename: str):
        self.filename = filename
        # origin data extract from nm tools
        self.nm_info = []
        # processed data with function information
        self.proc_info = []
        # aggregated data by its namespace and class
        # other filter method will add to here
        self.agg_info = []

    def fetch(self):
        mangle_lines = os.popen("nm -Dn " + self.filename).readlines()
        demangle_lines = os.popen("nm -DCn " + self.filename).readlines()

        nm_info = self.extractor(mangle_lines, demangle_lines)

        # 不解析virtual修饰的函数
        self.nm_info = [item for item in nm_info if not (
                item['demangle'].startswith('typeinfo ') or
                item['demangle'].startswith('vtable ') or
                item['demangle'].startswith('VTT ')
        )]

        self.proc_info = self.process(nm_info)
        self.agg_info = self.aggre(self.proc_info)

    def gen(self):
        # write h file
        ## ToDO

        # write cpp file
        self.src = ''
        self.__src_join(template_include, 0)
        self.__src_join(template_define_VG, 0)

        for i, namespace in enumerate(self.agg_info.keys()):
            if i != 0:
                self.__src_join('\n', 0)
            symbols = self.agg_info[namespace]
            space_num = len(namespace)
            for i, item in enumerate(namespace):
                self.__src_join(template_namespace_entry.format(item), i)
            for i, symbol in enumerate(symbols):
                if i != 0:
                    self.__src_join('\n', 0)
                args_type = ', '.join(symbol['args'])
                args_decl = ', '.join([arg + ' a{:d}'.format(i) for i, arg in enumerate(symbol['args'])])
                args_list = ', '.join(['a{:d}'.format(i) for i in range(len(symbol['args']))])
                args_num = len(symbol['args'])
                ret = symbol['ret'] if len(symbol['ret']) != 0 else 'void'
                self.__src_join(template_func_pointer.format(ret, i, args_type), space_num)

                if symbol['constructor'] or symbol['constructor']:
                    self.__src_join(template_func_entry.format(
                        symbol['ret'],
                        self.stack('', symbol['fun_class'], symbol['fun_name']),
                        args_decl
                    ), space_num)
                else:
                    self.__src_join (template_func_entry.format(
                        ret,
                        self.stack('', symbol['fun_class'], symbol['fun_name']),
                        args_decl
                    ), space_num)
                self.__src_join(template_debug_load, space_num + 1)
                self.__src_join(template_debug_entry.format(symbol['mangle'], symbol['fun_name']), space_num + 1)
                self.__src_join(template_debug_save, space_num + 1)
                self.__src_join(template_dlsym.format(i, i, symbol['mangle']), space_num + 1)
                self.__src_join(template_dlsym_error.format(st='\t' * (space_num + 1)), space_num + 1)
                if symbol['ret'] == '':
                    self.__src_join(template_exec_no_ret.format(args_list), space_num + 1)

                else:
                    self.__src_join(template_exec_with_ret.format(symbol['ret'], args_list), space_num + 1)
                    self.__src_join(template_exec_ret, space_num + 1)

                self.__src_join(template_debug_load, space_num + 1)
                self.__src_join(template_debug_exit, space_num + 1)
                self.__src_join(template_debug_save, space_num + 1)
                self.__src_join(template_func_exit, space_num)
            for i, item in enumerate(namespace):
                self.__src_join(template_namespace_exit, space_num - i - 1)

    def __src_join(self, addon, tabs):
        self.src += '\t' * tabs
        self.src += addon
        self.src += '\n'

    @staticmethod
    def extractor(mangle_lines: list, demangle_lines: list):
        all_list = []
        for lineno, l in enumerate(mangle_lines):
            line_list = {}

            type_pos = re.search('[ ][A-Za-z][ ]', l).span()[0] + 1
            # symbol value
            if l[0] == " ":
                line_list['value'] = ""
            else:
                line_list['value'] = l[:type_pos - 1]
            # symbol type
            line_list['type'] = l[type_pos]
            # mangle name
            line_list['mangle'] = l[type_pos + 2:-1]
            # demangle name
            line_list['demangle'] = demangle_lines[lineno][type_pos + 2:-1]

            all_list.append(line_list)
        return all_list

    @staticmethod
    def stack(space: str, cls: str, func: str):
        outerult = space
        if len(space) > 0:
            outerult += '::'
        if cls == '':
            return func
        return outerult + cls + '::' + func

    @staticmethod
    def balance_split(origin: str, flag: str):
        balance = 0
        parts = []
        target = origin
        end = False
        leng = len(flag)

        while not end:
            end = True
            for i in range(len(target)):
                if target[i] == '<':
                    balance += 1
                elif target[i] == '>':
                    balance -= 1
                elif target[i:i + leng] == flag and balance == 0:
                    parts.append(target[:i])
                    target = target[i + leng:]
                    end = False
                    break

        parts.append(target)
        return tuple(parts)

    @classmethod
    def process(cls, nm_info: list):
        mode = re.compile('^(.*)?\((.*)\)( const)?$', re.S)
        # unsigned 用于处理<unsigned long>
        # , 用于处理<char, std::char_traits<char>>
        # operator 用于处理operator new
        # const 用于处理<int const>
        # 以上三处不可从中间切断
        mode_inner = re.compile('^(.*(?<!unsigned)(?<!,)(?<!operator) )?((?!const).*)$', re.S)

        for item in nm_info:
            # flag预设为false
            item['cstyle'] = False
            item['other'] = False
            item['constructor'] = False
            item['destructor'] = False
            item['static'] = False

            raw = item['demangle']
            mid = raw.replace('> ', '>')
            # mid, units, start = extract_bracket(raw)

            outer = mode.match(mid)

            # import的函数(不显示args)
            if outer is None:
                item['fun_name'] = item['demangle']
                item['fun_class'] = ''
                item['fun_space'] = tuple()
                item['args'] = tuple()
                item['ret'] = ''
                item['cstyle'] = True
                continue

            tmp_pre = outer.group(1)
            inner = mode_inner.match(tmp_pre)
            ret = inner.group(1)

            func_pre = cls.balance_split(inner.group(2), '::')

            args = cls.balance_split(outer.group(2), ', ')

            # const函数
            if outer.group(3) is None:
                item['const'] = False
            else:
                item['const'] = True

            # 无返回值
            item['ret'] = ret if ret else ''

            item['fun_name'] = func_pre[-1]

            # 单函数
            if len(func_pre) == 1:
                item['fun_class'] = ''
                item['fun_space'] = tuple()
            # 类函数但没有名称空间
            elif len(func_pre) == 2:
                item['fun_class'] = func_pre[-2]
                item['fun_space'] = tuple()
            # 类函数有名称空间
            else:
                item['fun_class'] = func_pre[-2]
                item['fun_space'] = tuple(func_pre[:-2])

            # 没有参数的函数
            item['args'] = tuple() if (len(args) == 1 and len(args[0]) == 0) else tuple(args)

            # 构造方法
            if item['fun_name'] == item['fun_class']:
                item['constructor'] = True
            # 析构方法
            if item['fun_name'] == '~' + item['fun_class']:
                item['destructor'] = True
            # 从其他lib引入的非系统符号
            if item['type'] == 'U' and outer:
                item['other'] = True
            # 可能是静态方法(未知)
            if len(item['args']) > 0 and (
                    cls.stack('::'.join(item['fun_space']), item['fun_class'], item['fun_name']) == item['args'][
                0] or cls.stack('::'.join(item['fun_space']), item['fun_class'], item['fun_name']) == item['args'][
                        0] + '&'):
                item['static'] = True

            item['flag'] = 6 if item['cstyle'] else 0 + 8 if item['other'] else 0 + 4 if item[
                'constructor'] else 0 + 2 if item['destructor'] else 0 + 1 if item['static'] else 0

        return nm_info

    @staticmethod
    def aggre(proc_info: list):
        agg_info = {}
        proc_info = [item for item in proc_info if not item['cstyle']]
        proc_dict = {item['demangle']: item for item in proc_info}

        for key in proc_dict.keys():
            symbol = proc_dict[key]
            if symbol['fun_space'] not in agg_info.keys():
                agg_info[symbol['fun_space']] = [symbol]
            else:
                agg_info[symbol['fun_space']].append(symbol)
        return agg_info
