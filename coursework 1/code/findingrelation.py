class FindRelation(): # define a new class called FindRelation
    def __init__(self): # initialize parameters
        self.count1 = 0 # define 2 counters
        self.count2 = 0

    def counter(self, data1, dataset1, data2= [], dataset2=[]): # define a method called counter to count the number of dataset 2 in dataset 1
        self.__init__() # initialize the parameters
        for index in range(len(data1)):
            if data1[index] in dataset1:
                self.count1 += 1 # count the number of dataset1 in data1 as the condition
                if len(data2) != 0 : # if data2 is not empty
                    if data2[index] in dataset2:
                        self.count2 += 1 # count number of data2 in dataset2 under the condition dataset1
        return [self.count1, self.count2] # return two counts


    def calculate_probabilities(self, data1, dataset1, data2, dataset2): # define a method to compute conditional probability
        '''target probability'''
        if len(dataset1) == 0: # if condition is empty, return 0
            return 0

        else:
            # conditional probability = count2 / count1
            p_dataset1_dataset2 = self.counter(data1, dataset1, data2, dataset2)[1] / self.counter(data1, dataset1, data2, dataset2)[0]
            return round(p_dataset1_dataset2, 4) # return probability with 4 decimals


    # def calculate_probabilities(self, data1, dataset1, data2, dataset2):
    #     p_dataset1 = self.counter(data1, dataset1)[0]/ len(data1)
    #     p_dataset2 = self.counter(data2, dataset2)[0] / len(data2)
    #     p_dataset2_dataset1 = self.counter(data1, dataset1, data2, dataset2)[1]/self.counter(data1, dataset1, data2, dataset2)[0]
    #     p_dataset1_dataset2 = p_dataset2_dataset1 * p_dataset1/p_dataset2 # target probability
    #     return round(p_dataset1_dataset2, 4)

