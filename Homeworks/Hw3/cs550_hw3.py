import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, accuracy_score, fbeta_score


class Genetic:
    def __init__(self, num_population, num_generations, prob_mutation=0.4, survive_ratio=0.5):

        self.num_population = num_population
        self.num_generations = num_generations
        self.prob_mutation = prob_mutation
        self.survive_ratio = survive_ratio
        self.num_survive = 0

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

    def selected_features(self, dna, train_x, test_x):
        """
        Return a training set that has features selected by the respective DNA.

        :return: A list that stores features determined by each dna
        """

        selected_features_train = train_x[:, np.argwhere(dna == 1)]
        selected_features_train = np.squeeze(selected_features_train)
        selected_features_test = test_x[:, np.argwhere(dna == 1)]
        selected_features_test = np.squeeze(selected_features_test)

        return selected_features_train, selected_features_test

    def fitness(self, fbeta, feature_cost_list, dna):
        """
        Return a list of fitness values with respect to each DNA.

        :param fbeta_scores: F_beta score of svm classifier that uses certain features selected by the respective DNA.
        :param feature_cost_list: Feature costs are taken into account for cost sensitive learning.
        :param population: Matrix of current population.
        :return: Fitness values of each DNA in the population.
        """

        subtract_cost = 0
        feat_cost = 0

        if dna[-1] == 1:
            if dna[-2] == 1 and dna[-3] == 0:
                subtract_cost += feature_cost_list[-2]
            elif dna[-2] == 0 and dna[-3] == 1:
                subtract_cost += feature_cost_list[-3]
            elif dna[-2] == 0 and dna[-3] == 0:
                subtract_cost = 0

        feat_cost += np.sum(feature_cost_list[np.argwhere(dna == 1)]) - subtract_cost
        #feat_cost = feat_cost/np.sum(feature_cost_list)  # Normalize feature cost

        fitness_val = np.exp(fbeta*7) / feat_cost
        return fitness_val


    def crossover(self, population, fit_val):
        """
        Cross-over the survived genes.
        Mask used: 111000111000111000111
        :param population:
        :return:
        """
        # print('Crossover method-------------')
        num_pop_to_crossover = int(np.round(self.num_population-self.num_survive))#*(1-self.survive_ratio)))

        if num_pop_to_crossover % 2 == 1:
            num_pop_to_crossover -= 1


        selection_prob = np.squeeze(fit_val / np.sum(fit_val))
        # print('crossover selection probs: \n', selection_prob)
        crossover_idx = np.random.choice(len(population), num_pop_to_crossover, replace=False, p=selection_prob)
        # print('Crossover olmaya secilen idx: ', crossover_idx)
        crossover_population = population[crossover_idx, :]

        # print('Pop to be crossover:\n', crossover_population)

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
        # print('Mutate method----------')
        mutated_gen = []
        # DNA selection to be mutated can be improved by selecting DNAs whose fitness values are lower.
        num_pop_to_mutate = int(np.round(self.num_population*self.prob_mutation))

        if num_pop_to_mutate == 0:
            return evolved_population

        mutate_idx = np.random.choice(len(evolved_population), num_pop_to_mutate, replace=False)
        # print('IDX to mutate: ', mutate_idx)
        mutate_dnas = evolved_population[mutate_idx, :]
        # print('dnas to be mutated:\n', mutate_dnas)

        # Mutate a random bit of each dna in selected portion of population (m.p individuals).
        current_dna_idx = 0
        for dna in mutate_dnas:
            mutate_bit_idx = np.random.randint(len(dna))
            # print('MUTATE BIT: ', mutate_bit_idx)
            dna[mutate_bit_idx] = np.bitwise_xor(dna[mutate_bit_idx], 1)
            evolved_population[mutate_idx[current_dna_idx]] = dna
            current_dna_idx += 1

        return evolved_population

    def next_generation(self, population, fit_val):
        """
        Next generation is selected probabilistically, with higher probability as higher fitness values.

        """
        # print('Next generation method----------')
        # fit_val = np.array(self.fitness_values)
        num_survive = int(np.round(self.num_population*self.survive_ratio))
        self.num_survive = num_survive

        if num_survive % 2 == 1:
            num_survive += 1

        selection_prob = np.squeeze(fit_val / np.sum(fit_val))
        # print('Selection probs: \n', selection_prob)
        # print('sum of probs: ', np.sum(selection_prob))
        # print('POPULATION: \n', population)
        # print('popsize: {}, prob: {}, and its size: {}'.format(population.shape[0], selection_prob, selection_prob.size))
        survive_idx = np.random.choice(population.shape[0], num_survive, replace=False, p=selection_prob)
        # print('survive idx: ', survive_idx)

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

    accuracy = np.round(accuracy_score(true_labels, predicted), decimals=3)
    if cm is False:
        return accuracy
    else:
        cm = confusion_matrix(true_labels, predicted)
        cls_based_accuracies = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cls_based_accuracies = np.round(cls_based_accuracies.diagonal(), decimals=3)
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


def run_genetic(genetic_config, wclf):

    population = genetic_config.init_population()
    #print('Initial population:\n', population)

    fit_vals = np.zeros((genetic_config.num_population, 1))
    fittest_value_record = np.zeros((genetic_config.num_generations, 1))

    fbeta_vals = np.zeros((genetic_config.num_population, 1))
    fittest_fbeta_record = np.zeros((genetic_config.num_generations, 1))

    cls_bsd_rec = []
    best_dna_record = np.zeros((genetic_config.num_generations, 21), dtype=np.int)
    best_acc_dna = []

    for i in range(genetic_config.num_generations):
        print('GENERATION: {} ------------------------------'.format(i))
        print('Population:\n', population)

        # HERE TRAIN SVM ACC. TO SELECTED FEATURES
        # PASS F1 SCORES FOR EACH SVM TRAINED ON FEATURES THAT ARE SELECTED BY DNAs.
        for j in range(genetic_config.num_population):
            selected_x_train, selected_x_test = genetic_config.selected_features(population[j], train_x, test_x)
            # selected_x_train = normalize(selected_x_train)
            # selected_x_test = normalize(selected_x_test)

            wclf.fit(selected_x_train, train_y)

            predicted = wclf.predict(selected_x_train)
            print('PREDICTED includes: ', np.unique(predicted))
            #print('Test_Y includes: ', np.unique(train_y))

            f_beta = fbeta_score(train_y, predicted, beta=1.1, average='weighted')
            fbeta_vals[j] += f_beta

            acc, cm, cls_based = evaluate(train_y, predicted)
            cls_bsd_rec.append(cls_based)

            if np.min(cls_based) > 0.83:
                best_acc_dna.append(population[j])
                print('Good DNA recorded. Index: ', len(cls_bsd_rec)-1)

            # class_based_acc_record[j] += cls_based
            # print('DNA{} F_beta score: {}'.format(j, f_beta))

            fitness_val = genetic_config.fitness(f_beta, feature_costs, population[j])
            # print('DNA{} fitness: {}'.format(j, fitness_val))
            fit_vals[j] += fitness_val

        fittest_value_record[i] += np.max(fit_vals)
        fittest_fbeta_record[i] += fbeta_vals[np.argmax(fit_vals)]
        best_dna_record[i] += population[np.argmax(fit_vals)]

        print('\nFittest value: ', fittest_value_record[i])
        print('Fittest FBETA value: {}\n'.format(fittest_fbeta_record[i]))
        # print('Fittest Class-based acc: ', class_based_acc_record[np.argmax(fit_vals)])

        # Create new generation
        survived_dna = genetic_config.next_generation(population, fit_vals)
        # print('direct Next GEN: \n', survived_dna)
        crossed_dna = genetic_config.crossover(population, fit_vals)
        # print('Crossed:\n', crossed_dna)
        next_gen = np.concatenate((survived_dna, crossed_dna), axis=0)
        # cp_next = np.copy(next_gen)
        # print('NEXT GEN: \n', next_gen)
        next_gen = genetic_config.mutate(next_gen)
        # print('NEXT GEN: \n', next_gen)
        population = next_gen

        if i == genetic_config.num_generations-1:
            last_best_idx = np.argmax(fit_vals)
        # Zero-out fitness values for the new generation.
        fit_vals = np.zeros((gen.num_population, 1))
        fbeta_vals = np.zeros((gen.num_population, 1))

    return fittest_value_record, fittest_fbeta_record, best_dna_record, cls_bsd_rec, last_best_idx, population, best_acc_dna

#%% MAIN

train_x = np.loadtxt('./cs550_hw3_data/ann-train.txt')[:, :-1]
train_y = np.loadtxt('./cs550_hw3_data/ann-train.txt')[:, -1].astype(int)
test_x = np.loadtxt('./cs550_hw3_data/ann-test.txt')[:, :-1]
test_y = np.loadtxt('./cs550_hw3_data/ann-test.txt')[:, -1].astype(int)
feature_costs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0, 22.78, 11.41, 14.51, 11.41, 25.92])


#%% CLASSIFIER: class-weighted SVM

cls1_w, cls2_w, cls3_w = calc_weighted_costs(train_y)

# Fit and train the model using weighted classes
wclf = svm.SVC(C=6, kernel='linear', gamma='auto', class_weight={1: cls1_w, 2: cls2_w, 3: cls3_w})

#%% GENETIC
gen = Genetic(num_population=6, num_generations=21, prob_mutation=0.3, survive_ratio=0.4)
fittest_values, fbeta_record, best_dnas, \
class_based_acc_record, last_best_dna_idx, last_generation, best_acc_dna = run_genetic(gen, wclf)

print('Last Generation: \n', last_generation)
print('Last best dna index: ', last_best_dna_idx)
print('Fittest Vals:\n{}'.format(fittest_values))
print('Fbeta Scores: \n{}'.format(fbeta_record))
print('Last best class-based acc: ', class_based_acc_record[-gen.num_population:][last_best_dna_idx])

plt.plot(fittest_values)
plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.title('Fittest Value per Generation')
plt.show()

plt.plot(fbeta_record)
plt.ylabel('Fbeta')
plt.xlabel('Generation')
plt.title('Fbeta Score per Generation')
plt.show()

class_based_acc_record = np.array(class_based_acc_record)
x = np.arange(len(class_based_acc_record))
yc1 = class_based_acc_record[:,0]
yc2 = class_based_acc_record[:,1]
yc3 = class_based_acc_record[:,2]
y_array = np.array([yc1, yc2, yc3])
labels = ['class1', 'class2', 'class3']

for y_arr, label in zip(y_array, labels):
    plt.plot(x, y_arr, label=label)

plt.legend()
plt.title('Class-based Fbeta Scores')
plt.xlabel('DNAs (all generations)')
plt.ylabel('Fbeta Score')
plt.show()
