from fpy2 import fpy

def foo(n):
  @fpy()
  def bar(x):
    return x + n
  return bar

x = foo(0)
