from astropy.io import fits
import ls1p
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with fits.open("ccd.fits.gz") as fits_file:
        data = fits_file[0].data.astype(np.int16)

    u0 = np.mean(data[0,0,...])

    x = np.mean(data[:,0,...],
                axis = (1,2)) - u0
    sigma_Delta_x = np.mean(data[:, 0, ...] - data[:, 1, ...],
                axis=(1, 2))
    A = np.column_stack((x, np.ones_like(x)))
    xi, cost, var = lsp.lstsq_ne(A, sigma_Delta_x)
    print(xi)

    k, b = xi[0], xi[1]
    plt.plot(x, k*x+b)
    plt.scatter(x, sigma_Delta_x)


    print(A)
    print(x)
    print(sigma_Delta_x)
    plt.show()

