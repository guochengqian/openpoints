import hashlib
import json
import os
from ast import literal_eval
from typing import Any, Dict, List, Tuple, Union
from multimethod import multimethod
import yaml
import logging 


def print_args(args, printer=logging.info):
    printer("==========       args      =============")
    for arg, content in args.__dict__.items():
        printer("{}:{}".format(arg, content))
    printer("==========     args END    =============")

# 声明一个配置解析类，继承dict，dict是python内置的字典类
class EasyConfig(dict):
    def __getattr__(self, key: str) -> Any:
        """__getattr__方法
        这是一个 属性访问 的特殊方法。
        当访问一个不存在的属性时, Python 会调用这个方法。
        在 EasyConfig 类中，这个方法尝试从字典中获取对应的键值对。如果键不存在，则抛出 AttributeError 异常。
        """
        if key not in self:
            raise AttributeError(key)
        return self[key]
    # 
    def __setattr__(self, key: str, value: Any) -> None:
        """__setattr__方法
        这是一个 属性赋值 的特殊方法。
        当通过点操作符设置一个属性值时, Python 会调用这个方法。
        在 EasyConfig 类中，这个方法将属性的设置操作映射到字典的键值对设置上。
        """
        self[key] = value

    def __delattr__(self, key: str) -> None:
        """__delattr__方法
        这是一个 属性删除 的特殊方法。
        当通过 del 操作符删除一个属性时, Python 会调用这个方法。
        在 EasyConfig 类中，这个方法将属性的删除操作映射到字典的键值对删除上。"""
        del self[key]

    def load(self, fpath: str, *, recursive: bool = False) -> None:
        """定义一个load方法, 从yaml文件中加载配置
        Args:
            fpath (str): yaml文件的路径
            recursive (bool, optional): 是否递归加载其父默认yaml文件. Defaults to False.
        """
        # 如果fpath路径不存在，抛出FileNotFoundError异常
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        # eg: fpath = 'cfgs/s3dis/pointnet.yaml'
        fpaths = [fpath]
        # 如果recursive为True, 递归加载其父默认yaml文件
        if recursive:
            """递归示例
            1、cfgs/s3dis/pointnet.yaml
            2、cfgs/s3dis/default.yaml
            3、cfgs/default.yaml
            4、default.yaml
            """
            # os.path.splitext(fpath)返回一个元组，第一个元素是文件名，第二个元素是文件扩展名
            extension = os.path.splitext(fpath)[1]  # extension = '.yaml'
            # 检查是否到达根目录，若到达则停止循环
            while os.path.dirname(fpath) != fpath:  # 当父目录路径不等于fpath时
                # 将父目录路径传递给fpath
                fpath = os.path.dirname(fpath)
                # 将父目录路径下的default.yaml文件路径添加到fpaths列表中
                fpaths.append(os.path.join(fpath, 'default' + extension))
        '''fpaths列表示例
        ['cfgs/s3dis/pointnet.yaml', 'cfgs/s3dis/default.yaml', 'cfgs/default.yaml', 'default.yaml']
        '''
        # 逆序遍历fpaths列表
        for fpath in reversed(fpaths):
            # 如果fpath路径存在
            if os.path.exists(fpath):
                # 打开fpath路径的文件
                with open(fpath) as f:
                    # 一个文件对象 f 中读取 YAML 格式的数据，并调用update方法将其更新到当前对象的属性中
                    self.update(yaml.safe_load(f))

    def reload(self, fpath: str, *, recursive: bool = False) -> None:
        self.clear()
        self.load(fpath, recursive=recursive)

    # mutimethod makes python supports function overloading
    # mutimethod使python支持函数重载
    @multimethod
    def update(self, other: Dict) -> None:
        for key, value in other.items():
            if isinstance(value, dict):
                if key not in self or not isinstance(self[key], EasyConfig):
                    self[key] = EasyConfig()
                # recursively update
                self[key].update(value)
            else:
                self[key] = value

    @multimethod
    def update(self, opts: Union[List, Tuple]) -> None:
        index = 0
        while index < len(opts):
            opt = opts[index]
            if opt.startswith('--'):
                opt = opt[2:]
            if '=' in opt:
                key, value = opt.split('=', 1)
                index += 1
            else:
                key, value = opt, opts[index + 1]
                index += 2
            current = self
            subkeys = key.split('.')
            try:
                value = literal_eval(value)
            except:
                pass
            for subkey in subkeys[:-1]:
                current = current.setdefault(subkey, EasyConfig())
            current[subkeys[-1]] = value

    def dict(self) -> Dict[str, Any]:
        configs = dict()
        for key, value in self.items():
            if isinstance(value, EasyConfig):
                value = value.dict()
            configs[key] = value
        return configs

    def hash(self) -> str:
        buffer = json.dumps(self.dict(), sort_keys=True)
        return hashlib.sha256(buffer.encode()).hexdigest()

    def __str__(self) -> str:
        texts = []
        for key, value in self.items():
            if isinstance(value, EasyConfig):
                seperator = '\n'
            else:
                seperator = ' '
            text = key + ':' + seperator + str(value)
            lines = text.split('\n')
            for k, line in enumerate(lines[1:]):
                lines[k + 1] = (' ' * 2) + line
            texts.extend(lines)
        return '\n'.join(texts)