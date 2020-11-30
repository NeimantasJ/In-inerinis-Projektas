import numpy as np
import math
from multiprocessing import Pool
import multiprocessing
import json
import time

def read_file(filename):
  with open(filename) as json_file:
    data = json.load(json_file)
    for p in range(len(data["points"])):
      data["points"][p] = np.array([p, data["points"][p]["from"], data["points"][p]["to"]])
    return data['points']

def main(count):
  x = read_file('C:/Darbas/Studijos/Lygiagretusis/Projektas/datapack3.json')
  start_time = int(round(time.time() * 1000))
  p = Pool(count)
  p.map(greiciausio_nusileidimo, x)
  end_time = int(round(time.time() * 1000))
  print("Visas laikas (sec.) : ", (end_time - start_time) / 1000)
  
def greiciausio_nusileidimo(x):
  eps = 1e-10
  step = 0.1
  step0= 0.1
  itmax = 1000
  index = x[0]
  start_x = np.array([x[1], x[2]])
  x = np.array([x[1], x[2]])
  for i in range(1, itmax + 1):
      grad = gradient(x)
      fff = target(x)
      for j in range(1, 30):
        deltax = grad / np.linalg.norm(grad) * step

        arr = [x[j] - float(np.transpose(deltax)[j][0]) for j in range(len(x))]
        x = np.array(arr)

        fff1 = target(x)
        if fff1 > fff:
          arr = [x[j] + float(np.transpose(deltax)[j][0]) for j in range(len(x))]
          x = np.array(arr)
          step /= 10
        else:
          fff = fff1   
      step = step0

      tikslumas = np.linalg.norm(fff)
      #print('iteracija %d  tikslumas %g' %(i,tikslumas));
      if tikslumas < eps:
        print('NR. {0:.0f}, SPRENDINYS x1 = {1:.5f}, x2 = {2:.5f}'.format(index + 1, x[0], x[1]))
        break
      elif i == itmax:
        print('**** Tikslumas nepasiektas : ', start_x)
        break
  #end_time = int(round(time.time() * 1000))
  #print("Užtruko (sec.) : ", (end_time - start_time) / 1000)

def gradient(x):
  return np.dot(np.transpose(funk(x)), df(x))

def target(x):
  return (np.dot(np.transpose(funk(x)), funk(x)))/2

def funk(x):
  a = np.array(
    [[ (x[0]**2 + x[1]**2) / 5 - 2 * math.cos(x[0]/2) - 6 * math.cos(x[1]) - 8 ],
      [ (x[0]/2)**5 + (x[1]/2)**4 - 4 ]])
  a.shape = (2, 1)
  a = np.matrix(a).astype(np.float)
  return a

def df(x):
  s = np.array(
      [[ (2 * x[0])/5 + math.sin(x[0]/2), (2 * x[1]) / 5 + 6 * math.sin(x[1]) ],
    [ (5 * x[0]**4) / 32, (x[1]**3) / 4 ]]
  )

  s.shape = (2, 2)
  return np.matrix(s).astype(np.float)

if __name__ == '__main__':
    print("Number of cpu : ", multiprocessing.cpu_count())
    for i in range(1, 5):
      main(i)