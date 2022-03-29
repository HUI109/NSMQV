# 用來算指數亂數分佈
# import random
import math
import javarandom


class Generator:
    def __init__(self):
        # random = random.Random()
        # self.random_num = 0
        self.seed = 0
    #
    # def generator(self):
        self.random = javarandom.Random()
        self.random.setSeed(self.seed)

    def setSeed(self, seed):
        self.random.seed(seed)

    def getRandomNumber_Exponential(self, mean):
        return -(math.log(self.random.nextDouble()) * mean)

    def getRandomNumber_Poisson(self, nsmqv_lambda):
        L = math.exp(-nsmqv_lambda)
        k = 0
        p = 1.0
        while True:
            k += 1
            p = p * self.random.nextDouble()
            if p > L:
                break

        return k - 1

    def getRandomNumber_Uniform(self, mean):
        return self.random.nextDouble() * mean

    def getRandomNumber_Normal(self, mean):
        return abs(self.random.nextGaussian() * mean)
