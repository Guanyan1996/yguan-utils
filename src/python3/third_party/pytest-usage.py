import pytest

"""
conftest.py
根据函数test_$(name)动态返回不同结果。
会在函数生成执行tests之前，根据不同的函数名生成对应的变量
可以通过此方法动态生成sql 或者request 结果，不需要重复使用
理论上是在每个函数前生成pytest.mark.parametrize(pb,[])。
注意：如果是重复性一直使用的大量数据，不建议使用此方法，会导致内存占用过高，每个test函数都会调用一遍，产生较高的内存损耗。
可以使用： from functools import lru_cache的方式减少时间损耗

metafunc.fixturenames
"""


def pytest_generate_tests(metafunc):
    values = metafunc.config.getoption("--values")
    print("调用pytest_generate")
    if "attr" in metafunc.definition.name:
        metafunc.parametrize("key", values)
    elif "attr" in metafunc.definition.name:
        metafunc.parametrize("key", values)
    elif "attr" in metafunc.definition.name:
        metafunc.parametrize("key", values)
    else:
        metafunc.parametrize("key", values)


"""
# conftest.py
# 根据名称动态增加标签
# 使用方式 -m fid/did/common
# ------------------
"""


def pytest_configure(config):
    marker_list = ["a", "b", "c", "d"]  # 标签名集合
    for markers in marker_list:
        config.addinivalue_line(
            "markers", markers
        )


def pytest_collection_modifyitems(items):
    for item in items:
        if "a" in item.nodeid:
            item.add_marker(pytest.mark.a)
        elif "b" in item.nodeid:
            item.add_marker(pytest.mark.b)
        elif "c" in item.nodeid:
            item.add_marker(pytest.mark.c)
        else:
            item.add_marker(pytest.mark.d)


"""
启动pytest传自定义参数
pytest --abc -s -v test_example.py
Store_True的意思等同于--face_free调用为true,不调用则为default默认值
"""


def pytest_addoption(parser):
    parser.addoption(
        "--abc", default="", help=""
    )


@pytest.fixture(scope="session")
def abc(request):
    return request.config.getoption("--abc")
