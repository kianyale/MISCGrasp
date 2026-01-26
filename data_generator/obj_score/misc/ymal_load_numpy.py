import base64

import yaml
import numpy as np

b = {'A0': (np.random.random(1), {'fuck': 2}), 'B0': (np.random.random([3, 3]), 'fuck')}
with open('../data.yaml', 'w') as f:
    yaml.dump(b, f)

import yaml
import numpy as np

# 自定义构造器来解析 numpy 对象
def numpy_tuple_constructor(loader, node):
    value = loader.construct_sequence(node)
    return tuple(value)

def numpy_ndarray_constructor(loader, node):
    # 解析 `numpy.ndarray` 对象
    value = loader.construct_mapping(node)
    dtype = value.get('dtype', np.float64)  # 默认使用 float64 类型
    shape = value.get('shape', ())
    data = np.frombuffer(value.get('data', ''), dtype=dtype).reshape(shape)
    return data

def numpy_dtype_constructor(loader, node):
    # 解析 `numpy.dtype` 对象
    value = loader.construct_sequence(node)
    return np.dtype(value)

# 注册 YAML 构造器
yaml.add_constructor('!python/tuple', numpy_tuple_constructor)
yaml.add_constructor('!python/object/apply:numpy.core.multiarray._reconstruct', numpy_ndarray_constructor)
yaml.add_constructor('!python/object/apply:numpy.dtype', numpy_dtype_constructor)

# 读取并解析 YAML 文件
yaml_file = '../data.yaml'  # 这里是你的 YAML 文件路径

with open(yaml_file, 'r') as file:
    data = yaml.load(file, Loader=yaml.Loader)

# 输出解析结果
print(data)
