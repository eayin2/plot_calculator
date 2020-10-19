# Geometric series S = \sum_{i=0}^n q**n with |q| < 1 converges against a finite number
# Example with q = .2

repetitions = 50000
r = 0.0
for i in range(repetitions):
    ii = i+1
    r += (.2)**ii

print(r)
