from dataclasses import dataclass


@dataclass
class Person:
    name: str
    age: int
    gender: str


def func(person):  # person is instance of `Person` class
    match person:
        # This is not a constructor
        case Person(name, age, _) if age < 18:  # guard for extra filtering
            print(f"{name} is a child.")
        case Person(name=name, age=_, gender="male"):  # Wildcard ("throwaway" variable) can be used
            print(f"{name} is man.")
        case Person(name=name, age=_, gender="female"):
            print(f"{name} is woman.")
        case Person(name, age, _) if age < 100:  # Positional arguments work
            print(f"{name} is {age} years old.")
        case _:
            print(f"{person}")


func(Person("Lucy", 30, "female"))
# Lucy is woman.
func(Person("Ben", 15, "male"))
# Ben is a child.
func(Person("Ben", 122, "malea"))
