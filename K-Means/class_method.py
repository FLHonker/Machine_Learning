class method():

    def a(self):
        print(111)

    def b(self, c=a):
        c(self)

m = method()
m.b()