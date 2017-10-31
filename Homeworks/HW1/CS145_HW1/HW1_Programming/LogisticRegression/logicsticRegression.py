import math
import numpy
#-------------------------------------------------------------------
def log(n):
    return math.log(n)
#-------------------------------------------------------------------
def exp(n):
    return math.exp(n)
#-------------------------------------------------------------------
class logistic:
    #******************************************************
    def __init__(self, parameters):
        self.parameters = parameters
    #******************************************************
    ########## Feel Free to Add Helper Functions ##########
    #******************************************************
    def log_likelihood(self):
        ll = 0.0
        ##################### Please Fill Missing Lines Here #####################
        try:
            l0 = 1 + math.exp(self.parameters[0] + 60 * self.parameters[1] + 155 * self.parameters[2])
            l1 = 1 + math.exp(self.parameters[0] + 64 * self.parameters[1] + 135 * self.parameters[2])
            l2 = 1 + math.exp(self.parameters[0] + 73 * self.parameters[1] + 170 * self.parameters[2])
        except  OverflowError:
            ll = -self.parameters[0] - 60 * self.parameters[1] - 155 * self.parameters[2]
            return ll
        ll = 2 * self.parameters[0] + 137 * self.parameters[1] + 305 * self.parameters[2] - math.log(1 + math.exp(self.parameters[0] + 60 * self.parameters[1] + 155 * self.parameters[2])) - math.log(l1 = 1 + math.exp(self.parameters[0] + 64 * self.parameters[1] + 135 * self.parameters[2])) - math.log(1 + math.exp(self.parameters[0] + 73 * self.parameters[1] + 170 * self.parameters[2]))
        return ll
    #******************************************************
    def gradients(self):
        gradients = []
        ##################### Please Fill Missing Lines Here #####################
        try:
            l0 = 1 + math.exp(self.parameters[0] + 60 * self.parameters[1] + 155 * self.parameters[2])
        except OverflowError:
            l0 = float('inf')
        try:
            l1 = 1 + math.exp(self.parameters[0] + 64 * self.parameters[1] + 135 * self.parameters[2])
        except OverflowError:
            l1 = float('inf')
        try:
            l2 = 1 + math.exp(self.parameters[0] + 73 * self.parameters[1] + 170 * self.parameters[2])
        except OverflowError:
            l2 = float('inf')
        g0 = -1 + 1/l0 + 1/l1 + 1/l2
        g1 = -60 + 60/l0 + 64/l1 + 73/l2
        g2 = -155 + 155/l0 + 135/l1 + 170/l2
        gradients.append(g0)
        gradients.append(g1)
        gradients.append(g2)
        return gradients
    #******************************************************
    def iterate(self):
        ##################### Please Fill Missing Lines Here #####################
        try:
            self.parameters = self.parameters - numpy.linalg.inv(self.hessian()).dot(self.gradients())
        except:
            d = numpy.zeros(self.hessian().shape)
            for i in range(self.hessian().shape[1]):
                d[i, i] = 10**(-6)
            self.parameters = self.parameters - numpy.linalg.inv(self.hessian()+d).dot(self.gradients())
        return self.parameters
    #******************************************************
    def hessian(self):
        n = len(self.parameters)
        hessian = numpy.zeros((n, n))
        ##################### Please Fill Missing Lines Here #####################
        l0 = self.parameters[0] + 60 * self.parameters[1] + 155 * self.parameters[2]
        l1 = self.parameters[0] + 64 * self.parameters[1] + 135 * self.parameters[2]
        l2 = self.parameters[0] + 73 * self.parameters[1] + 170 * self.parameters[2]
        try:
            d0 = 2 + math.exp(l0) + math.exp(-l0)
        except OverflowError:
            d0 = float('inf')
        try:
            d1 = 2 + math.exp(l1) + math.exp(-l1)
        except OverflowError:
            d1 = float('inf')
        try:
            d2 = 2 + math.exp(l2) + math.exp(-l2)
        except OverflowError:
            d2 = float('inf')
        hessian[0] = [-1/d0 - 1/d1 - 1/d2, -60/d0 - 64/d1 - 73/d2, -155/d0 - 135/d1 - 170/d2]
        hessian[1] = [-60/d0 - 64/d1 - 73/d2, -3600/d0 - 4096/d1 - 5329/d2, -9300/d0 - 8460/d1 - 12410/d2]
        hessian[2] = [-155/d0 - 135/d1 - 170/d2, -9300/d0 - 8640/d1 - 12410/d2, -24025/d0 - 18225/d1 - 28900/d2]
        return hessian
#-------------------------------------------------------------------
parameters = []
##################### Please Fill Missing Lines Here #####################
parameters.append(0.25)
parameters.append(0.25)
parameters.append(0.25)
## initialize parameters
l = logistic(parameters)
parameters = l.iterate()
l = logistic(parameters)
print (l.iterate())