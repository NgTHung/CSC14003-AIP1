from problems import Rastrigin, Ackley

r = Rastrigin()
print(r.eval(r.sample(5)))
a = Ackley()
print(a.eval(a.sample(5)))