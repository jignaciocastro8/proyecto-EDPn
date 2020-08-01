from Person import *
class Student(Person):
    def __init__(self, nombre):
        super().__init__(nombre)



hola = Student("Hola")
hola.printNombre()