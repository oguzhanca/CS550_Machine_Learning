import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


class Genetic:
    def __init__(self, num_population, num_generations, prob_mutation, survive=0.5):

        self.num_population = num_population
        self.num_generations = num_generations
        self.prob_mutation = prob_mutation
        self.survive = survive

        self.population = np.zeros((self.num_population, 21), dtype=np.int)
        self.fitness_values = []

    def init_population(self):

        for i in range(self.num_population):
            individual = np.random.randint(2, size=21)
            self.population[i] += individual
        return self.population

    def selected_feature_list(self):
        selected_feature_list = []
        for i in range(self.num_population):
            selected_features = train_x[:, np.argwhere(self.population[i] == 1)]
            selected_features = np.squeeze(selected_features)
            selected_feature_list.append(selected_features)
        return selected_feature_list

    def fitness_list(self, f1_scores, feature_cost_list):
        self.fitness_values = []
        for p in self.population:
            add_cost = 0
            feat_cost = 0

            if p[-1] == 1:
                if p[-2] == 1 and p[-3] == 0:
                    add_cost += feature_cost_list[-3]
                elif p[-2] == 0 and p[-3] == 1:
                    add_cost += feature_cost_list[-2]
                elif p[-2] == 0 and p[-3] == 0:
                    add_cost += feature_cost_list[-1]

            feat_cost += np.sum(feature_cost_list[np.argwhere(p == 1)]) + add_cost
            feat_cost = feat_cost/np.sum(feature_cost_list)  # Normalize feature cost

            # fitness can be changed.....
            fitness_val = f1_scores / feat_cost
            self.fitness_values.append(fitness_val)
        return np.array(self.fitness_values)

    def crossover(self, survived_genes):
        """
        Cross-over the survived genes.
        Mask is: 111000111000111000111
        :param survived_genes:
        :return:
        """
        if len(survived_genes) % 2 == 1:
            print('Warning! Odd number of survived genes!')

        crossed_gen = []

        for i in range(0, len(survived_genes), 2):
            offspring1 = np.concatenate((survived_genes[i][:3], survived_genes[i+1][3:6], survived_genes[i][6:9],
                                         survived_genes[i+1][9:12], survived_genes[i][12:15], survived_genes[i+1][15:18],
                                         survived_genes[i][18:21]))
            offspring2 = np.concatenate((survived_genes[i+1][:3], survived_genes[i][3:6], survived_genes[i+1][6:9],
                                         survived_genes[i][9:12], survived_genes[i+1][12:15], survived_genes[i][15:18],
                                         survived_genes[i+1][18:21]))
            crossed_gen.append(offspring1)
            crossed_gen.append(offspring2)

        '''
        for i in range(0, len(survived_genes), 2):
            offspring1 = np.concatenate((survived_genes[i][:7], survived_genes[i+1][7:14], survived_genes[i][14:]))
            offspring2 = np.concatenate((survived_genes[i+1][:7], survived_genes[i][7:14], survived_genes[i+1][14:]))
            crossed_gen.append(offspring1)
            crossed_gen.append(offspring2)
        '''
        return np.array(crossed_gen)




    def next_generation(self):

        """
        Next generation is selected among the population according to the best fitness values.

        """
        fit_val = np.array(self.fitness_values)
        num_new_population = int(np.round(self.num_population*self.survive))

        if num_new_population % 2 == 1:
            num_new_population += 1

        selection_prob = np.squeeze(fit_val / np.sum(fit_val))
        print('Selection probs: ', selection_prob)
        print('sum of probs: ', np.sum(selection_prob))

        next_gen_idx = np.random.choice(len(self.population), num_new_population, replace=False, p=selection_prob)
        print('nextgen idx: ', next_gen_idx)

        next_generation = self.population[next_gen_idx, :]

        return next_generation


def evaluate(true_labels, predicted, cm=True):

    accuracy = accuracy_score(true_labels, predicted)
    if cm is False:
        return accuracy
    else:
        cm = confusion_matrix(true_labels, predicted)
        cls_based_accuracies = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return accuracy, cm, cls_based_accuracies


def calc_weighted_costs(Y):
    """
    Since thyroid dataset is quite imbalanced, assigning special weights to
    classes gives more accurate class-based classification performance.
    :param Y: train or test labels.
    :return: penalty weights of each class for SVM classification.
    """
    num_samples_cls3 = np.count_nonzero(np.argwhere(Y == 3))
    num_samples_cls2 = np.count_nonzero(np.argwhere(Y == 2))
    num_samples_cls1 = np.count_nonzero(np.argwhere(Y == 1))

    cls1_weight = (num_samples_cls2+num_samples_cls3) / (num_samples_cls1+num_samples_cls2+num_samples_cls3)
    cls2_weight = (num_samples_cls1+num_samples_cls3) / (num_samples_cls1+num_samples_cls2+num_samples_cls3)
    cls3_weight = (num_samples_cls1+num_samples_cls2) / (num_samples_cls1+num_samples_cls2+num_samples_cls3)
    return cls1_weight*100, cls2_weight*100, cls3_weight*100



#%% MAIN

train_x = np.loadtxt('./cs550_hw3_data/ann-train.txt')[:, :-1]
train_y = np.loadtxt('./cs550_hw3_data/ann-train.txt')[:, -1].astype(int)
test_x = np.loadtxt('./cs550_hw3_data/ann-test.txt')[:, :-1]
test_y = np.loadtxt('./cs550_hw3_data/ann-test.txt')[:, -1].astype(int)
feature_costs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 22.78, 11.41, 14.51, 11.41, 25.92])

'''
cls1_w, cls2_w, cls3_w = calc_weighted_costs(train_y)


# Fit and train the model using weighted classes
wclf = svm.SVC(kernel='linear', gamma='scale', class_weight={1: cls1_w, 2: cls2_w, 3: cls3_w})
wclf.fit(train_x, train_y)

# Predict labels
predicted = wclf.predict(train_x)

acc, cm, cls_based = evaluate(train_y, predicted)

# Now the normalize the diagonal entries
print('\nconfusion matrix:\n', cm)
print('class-based: {}\noverall acc: {}'.format(cls_based.diagonal(), acc))
print('kernel: {}, gamma: {}, shrinking: {}'.format(wclf.kernel, wclf.gamma, wclf.shrinking))

f1_score = f1_score(train_y, predicted)
'''

gen = Genetic(10, 4, 0.4)
gen.init_population()

selected_features = gen.selected_feature_list()

fit_vals = gen.fitness_list(0.5, feature_costs)
print('fitness: ', fit_vals)

next_gen = gen.next_generation()
print('Next GEN: \n', next_gen)

crossed_pop = gen.crossover(next_gen)
print('Crossed:\n', crossed_pop)

