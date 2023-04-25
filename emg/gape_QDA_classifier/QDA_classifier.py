from scipy.io import loadmat

# Load the gape algorithm results from Li et al. 2016
a = loadmat('QDA_nostd_no_first.mat')
a = a['important_coefficients'][0]

# Define a function that applies the QD algorithm to each individual movement,
# with x = interval and y = duration.
# Returns True or False based on if the QD evaluates to <0 or not


def QDA(x, y):
    return (a[0] + a[1]*x + a[2]*y + a[3]*x**2 + a[4]*x*y + a[5]*y**2) < 0
