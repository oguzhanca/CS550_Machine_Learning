import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


class Genetic:
    def __init__(self, num_population, num_generations, prob_mutation=0.4, survive_ratio=0.5):

        self.num_population = num_population
        self.num_generations = num_generations
        self.prob_mutation = prob_mutation
        self.survive_ratio = survive_ratio

        self.population = np.zeros((self.num_population, 21), dtype=np.int)
        self.fitness_values = []

    def init_population(self):
        """
        Initialize a random population.

        :return:
        """

        for i in range(self.num_population):
            individual = np.random.randint(2, size=21)
            self.population[i] += individual
        return self.population

    def selected_feature_list(self, population):
        """
        Return a training set that has features selected by the respective DNA.

        :return: A list that stores features determined by each dna
        """
        selected_feature_list = []
        for i in range(self.num_population):
            selected_features = train_x[:, np.argwhere(population[i] == 1)]
            selected_features = np.squeeze(selected_features)
            selected_feature_list.append(selected_features)
        return selected_feature_list

    def fitness_list(self, f1_scores, feature_cost_list, population):
        """
        Return a list of fitness values with respect to each DNA.

        :param f1_scores: F1 score of svm classifier that uses certain features selected by the respective DNA.
        :param feature_cost_list: Feature costs are taken into account for cost sensitive learning.
        :param population: Matrix of current population.
        :return: Fitness values of each DNA in the population.
        """
        self.fitness_values = []
        for p in population:#self.population:
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

    def crossover(self, population, fit_val):
        """
        Cross-over the survived genes.
        Mask used: 111000111000111000111
        :param population:
        :return:
        """
        num_pop_to_crossover = int(np.round(self.num_population*(1-self.survive_ratio)))

        if num_pop_to_crossover % 2 == 1:
            num_pop_to_crossover -= 1


        selection_prob = np.squeeze(fit_val / np.sum(fit_val))
        print('crossover selection probs: ', selection_prob)
        crossover_idx = np.random.choice(len(population), num_pop_to_crossover, replace=False, p=selection_prob)
        print('Crossover olmaya secilen idx: ', crossover_idx)
        crossover_population = population[crossover_idx, :]

        print('Pop to be crossover:\n', crossover_population)

        if num_pop_to_crossover % 2 == 1:
            print('Warning! Odd number of genes to crossover!')

        crossed_gen = []

        for i in range(0, len(crossover_population), 2):
            offspring1 = np.concatenate((crossover_population[i][:3], crossover_population[i+1][3:6], crossover_population[i][6:9],
                                         crossover_population[i+1][9:12], crossover_population[i][12:15], crossover_population[i+1][15:18],
                                         crossover_population[i][18:21]))
            offspring2 = np.concatenate((crossover_population[i+1][:3], crossover_population[i][3:6], crossover_population[i+1][6:9],
                                         crossover_population[i][9:12], crossover_population[i+1][12:15], crossover_population[i][15:18],
                                         crossover_population[i+1][18:21]))
            crossed_gen.append(offspring1)
            crossed_gen.append(offspring2)

        return np.array(crossed_gen)

    def mutate(self, evolved_population):
        """
        Point mutation used.

        :param evolved_population: Crossovered population.
        :return: Next generation that has mutated random individuals.
        """
        mutated_gen = []
        # DNA selection to be mutated can be improved by selecting DNAs whose fitness values are lower.
        num_pop_to_mutate = int(np.round(self.num_population*self.prob_mutation))
        mutate_idx = np.random.choice(len(evolved_population), num_pop_to_mutate, replace=False)
        print('IDX to mutate: ', mutate_idx)
        mutate_dnas = evolved_population[mutate_idx, :]
        print('dnas to be mutated:\n', mutate_dnas)

        # Mutate a random bit of each dna in selected portion of population (m.p individuals).
        current_dna_idx = 0
        for dna in mutate_dnas:
            mutate_bit_idx = np.random.randint(22)
            print('MUTATE BIT: ', mutate_bit_idx)
            dna[mutate_bit_idx] = np.bitwise_xor(dna[mutate_bit_idx], 1)
            evolved_population[mutate_idx[current_dna_idx]] = dna
            current_dna_idx += 1

        return evolved_population

    def next_generation(self, population, fit_val):
        """
        Next generation is selected probabilistically, with higher probability as higher fitness values.

        """
        #fit_val = np.array(self.fitness_values)
        num_survive = int(np.round(self.num_population*self.survive_ratio))

        if num_survive % 2 == 1:
            num_survive += 1

        selection_prob = np.squeeze(fit_val / np.sum(fit_val))
        print('Selection probs: ', selection_prob)
        print('sum of probs: ', np.sum(selection_prob))

        survive_idx = np.random.choice(len(population), num_survive, replace=False, p=selection_prob)
        print('survive idx: ', survive_idx)

        #next_generation = self.population[next_gen_idx, :]
        survived_dnas = population[survive_idx, :]

        return survived_dnas


def evaluate(true_labels, predicted, cm=True):
    """
    Evaluate the classifier.

    :param true_labels:
    :param predicted:
    :param cm:
    :return:
    """

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


#%% CLASSIFIER: class-weighted SVM
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

#%% GENETIC
gen = Genetic(8, 4, 0.4)
population = gen.init_population()

selected_features = gen.selected_feature_list(population)
# HERE TRAIN SVM ACC. TO SELECTED FEATURES
# PASS F1 SCORES FOR EACH SVM TRAINED ON FEATURES THAT ARE SELECTED BY DNAs.
F1_score = 0.5
fit_vals = gen.fitness_list(F1_score, feature_costs, population)
print('fitness: ', fit_vals)

# Create new generation
survived_dna = gen.next_generation(population, fit_vals)
print('direct Next GEN: \n', survived_dna)
crossed_dna = gen.crossover(population, fit_vals)
print('Crossed:\n', crossed_dna)
next_gen = np.concatenate((survived_dna, crossed_dna), axis=0)
#cp_next = np.copy(next_gen)
print('NEXT GEN: \n', next_gen)
next_gen = gen.mutate(next_gen)
print('Mutated: \n', next_gen)
#population = next_gen


# AGAIN TRAIN SVM WITH FEATURES SELECTED BY THE EVOLVED GENERATION. REPEAT UNTIL CONVERGENCE.

