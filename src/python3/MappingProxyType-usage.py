from types import MappingProxyType

# froze dict
dict_1 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
dict_2 = MappingProxyType(dict_1)
dict_2.update({6: 6})
print(dict_2)
