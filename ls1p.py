import numpy as np
#import  matplotlib.pyplot as plt импортировать
#from astropy.io import fits

a1 = np.asarray([[1,0],[0,1]])
b1 = np.asarray([2,2])

#минимизация решения полученного методом наименьших квадратов
def lstsq_ne(a,b):
    A = np.linalg.inv(np.dot(a.T, a))
    B = np.dot(a.T, b) #a.t - транспонированная
    x = np.dot(A,B)
    r = np.dot(a,x) - b # вектор невязки
    cost = np.dot(r.T, r) # квадрат нормы невязки
    var = cost / (a.shape[0] - a.shape[1]) #матрица ошибок решения, a.shape(0) - размер столбцов


    return x, cost, var #возвращаем кортеж данных

#то что должно быть в файле evaaal.py

A_test = np.random.normal(size=(500,20)) #случайная матрица 500х20
x_test = np.random.normal(size=(20, )) #случайный вектор параметров x размера 20
b_test = A_test @ x_test + 0.01 * np.random.normal(size=(500, )) # вектор b получается произведенеиме Ax, @ вместо np.dot, noise - шум

x, cost, var = lstsq_ne(A_test, b_test)

#plt.hist[x -x_test, range=[-0.01, 0.01], bins=5]
#plt.scatter[x, x_test]
#plt.show() нарисовать график после импорта


noise_level = 0.01
noise = np.random.normal(size=(10000, 500))

X = np.asarray([lstsq_ne(A_test, A_test@x_test + n*noise_level)[0] for n in noise])
X.shape


#with fits.open('ccd.fits.gz') as fits_file:
    #data = fits_file[0].data.astype(np.int16)
    #print(data.shape(), type(data))



#plt.hist(x[:,0], bins=20)
#plt.show() нарисовать график после импорта

#form scipy.stats import norm

#x = np.linspace(norm.ppf(0.01), loc=x_test[0],  scale = [var[0][0]] )
#plt.hist(x[:,0], bins=20)
#ax.plot[x, norm.pdf(x), 'r', lw = 5, alpha = 0.6, label = ' norm pdf']





