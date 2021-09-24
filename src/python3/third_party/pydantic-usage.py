"""
!/usr/bin/python3.7
-*- coding: UTF-8 -*-
Author: https://github.com/Guanyan1996
         ┌─┐       ┌─┐
      ┌──┘ ┴───────┘ ┴──┐
      │                 │
      │       ───       │
      │  ─┬┘       └┬─  │
      │                 │
      │       ─┴─       │
      │                 │
      └───┐         ┌───┘
          │         │
          │         │
          │         │
          │         └──────────────┐
          │                        │
          │                        ├─┐
          │                        ┌─┘
          │                        │
          └─┐  ┐  ┌───────┬──┐  ┌──┘
            │ ─┤ ─┤       │ ─┤ ─┤
            └──┴──┘       └──┴──┘
                神兽保佑
                代码无BUG!

"""
import re
from typing import Optional, Union

from pydantic import BaseModel, Field, validator


class Student(BaseModel):
    age: int
    sex: Union[int, bool]
    test: Optional[int]
    default: int = 3
    password: str = Field(alias="key")

    class Config:
        allow_mutation = False

    @validator("password")
    def password_rule(cls, password):
        def is_valid(password):
            if len(password) < 6 or len(password) > 20:
                return False
            if not re.search("[a-z]", password):
                return False
            if not re.search("[A-Z]", password):
                return False
            if not re.search("\d", password):
                return False
            return True

        if not is_valid(password):
            raise ValueError("password is invalid")
        return password


class Password(BaseModel):
    password: str

    class Config:
        min_anystr_length = 6  # 令Password类中所有的字符串长度均要不少于6
        max_anystr_length = 20  # 令Password类中所有的字符串长度均要不大于20


test = Student(age=1, sex=123, test=123, key="helloworld1232aASA")
test.age = 23
print(test)
