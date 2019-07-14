from numpy import *

def compute_error(c,m,points):
    #initilize the error
    totalError = 0.0
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        #compute difference , square it , add it total
        totalError +=(y-(m*x+c))**2
    #getting average corresponding to m & c values
    return totalError/float(len(points))


def step_gradient(c_current,m_current,points,learning_rate):

    #starting point for our gradient descent
    c_gradient = 0
    m_gradient = 0

    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        #derivation wrt c and m
        #computing partial derivative of error fun
        c_gradient += (2/len(points))*(y-(m_current*x + c_current))
        m_gradient += (2/len(points))*x*(y-(m_current*x  + c_current ))

    #update our c & m value using partial derivative
    new_c = c_current + learning_rate*c_gradient
    new_m = m_current + learning_rate*m_gradient

    return  new_c,new_m


def gradient_descent(points,starting_c,starting_m,learning_rate,num_iteration):
    # starting value of m and c
    m = starting_m
    c = starting_c

    for i in range(num_iteration):
        #update c and m for every iteration by performing gradient step
        c,m=step_gradient(c,m,array(points),learning_rate)
    return [c,m]



def run():
    print("Function Run")
    print("Step 1 : Collect Data")
    points = genfromtxt('data.csv', delimiter=',')
    #Step 2 define our hyperparameter(what operations perform on data,analysing of data, speed of data)
    #how fast a model converge used in gradient descent algo
    learning_rate = 0.0001

    # equation of line m=slope, c=intersection on y-axis y=mx + c
    init_m = 0
    init_c = 0
    #how much time you want to train your dataset
    num_iteration = len(points)

    #Step : Train our model

    print('starting at c={0},m={1},error={2}'.format(init_c,init_m,compute_error(init_c,init_m,points)))
    [c,m] = gradient_descent(points,init_c,init_m,learning_rate,num_iteration)

    print('ending point at c={1},m={2},error={3}'.format(num_iteration,c,m,compute_error(c,m,points)))


if __name__ == '__main__':
    run()
