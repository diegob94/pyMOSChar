from typing import Dict
import numpy as np
from dataclasses import dataclass

@dataclass
class TestClass:
    my_dict: Dict[str, np.ndarray]
    def __post_init__(self):
        self.my_dict = {k:np.array(v) for k,v in self.my_dict.items()}

foo = TestClass(my_dict={'aa':[1,2,3,4]})
print(foo.my_dict,type(foo.my_dict))
print(foo.my_dict['aa'],type(foo.my_dict['aa']))
