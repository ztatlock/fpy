from fpy2 import fpy

#(FPCore (a b c)
# :name "NMSE p42, positive"
# :cite (hamming-1987 herbie-2015)
# :fpbench-domain textbook
# :pre (and (>= (* b b) (* 4 (* a c))) (!= a 0))
# (/ (+ (- b) (sqrt (- (* b b) (* 4 (* a c))))) (* 2 a)))

@fpy
def hamming_quad_positive(a, b, c):
    return (b - sqrt(b**2 - 4 * a * c)) / (2 * a)

print(hamming_quad_positive(10, 1, 1))
