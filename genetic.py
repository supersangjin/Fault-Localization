#
# genetic programming
#
import operator
import math
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


metrics = ["ochiai", "jaccard", "gp13", "wong1", "wong2", "wong3", "tarantula", "ample", "RussellRao", "SorensenDice",
           "Kulczynski1", "SimpleMatching", "M1", "RogersTanimoto", "Hamming", "ochiai2", "Hamann", "dice",
           "Kulczynski2", "Sokal", "M2", "Goodman", "Euclid", "Anderberg", "Zoltar", "ER1a", "ER1b", "ER5a", "ER5b",
           "ER5c", "gp02", "gp03", "gp19", "min_age", "max_age", "mean_age", "num_args", "num_vars", "b_length", "loc",
           "churn"]


def div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def sqrt(n):
    return math.sqrt(math.fabs(n))


pset = gp.PrimitiveSet("MAIN", 41)  # 33 + 8
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(sqrt, 1)
pset.addTerminal(1)

pset.renameArguments(ARG0=metrics[0])
pset.renameArguments(ARG1=metrics[1])
pset.renameArguments(ARG2=metrics[2])
pset.renameArguments(ARG3=metrics[3])
pset.renameArguments(ARG4=metrics[4])
pset.renameArguments(ARG5=metrics[5])
pset.renameArguments(ARG6=metrics[6])
pset.renameArguments(ARG7=metrics[7])
pset.renameArguments(ARG8=metrics[8])
pset.renameArguments(ARG9=metrics[9])
pset.renameArguments(ARG10=metrics[10])
pset.renameArguments(ARG11=metrics[11])
pset.renameArguments(ARG12=metrics[12])
pset.renameArguments(ARG13=metrics[13])
pset.renameArguments(ARG14=metrics[14])
pset.renameArguments(ARG15=metrics[15])
pset.renameArguments(ARG16=metrics[16])
pset.renameArguments(ARG17=metrics[17])
pset.renameArguments(ARG18=metrics[18])
pset.renameArguments(ARG19=metrics[19])
pset.renameArguments(ARG20=metrics[20])
pset.renameArguments(ARG21=metrics[21])
pset.renameArguments(ARG22=metrics[22])
pset.renameArguments(ARG23=metrics[23])
pset.renameArguments(ARG24=metrics[24])
pset.renameArguments(ARG25=metrics[25])
pset.renameArguments(ARG26=metrics[26])
pset.renameArguments(ARG27=metrics[27])
pset.renameArguments(ARG28=metrics[28])
pset.renameArguments(ARG29=metrics[29])
pset.renameArguments(ARG30=metrics[30])
pset.renameArguments(ARG31=metrics[31])
pset.renameArguments(ARG32=metrics[32])
pset.renameArguments(ARG33=metrics[33])
pset.renameArguments(ARG34=metrics[34])
pset.renameArguments(ARG35=metrics[35])
pset.renameArguments(ARG36=metrics[36])
pset.renameArguments(ARG37=metrics[37])
pset.renameArguments(ARG38=metrics[38])
pset.renameArguments(ARG39=metrics[39])
pset.renameArguments(ARG40=metrics[40])

creator.create("FitnessMin", base.Fitness, weights=(-5.0, -1.0))  # Fitness minimize
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=8)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=8)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))


def evalFitness(individual, dataset):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    total = []
    for input_class in dataset.input_data:
        method_rank = []
        for method in input_class:
            method.score = func(*method.method_data)
            method_rank.append(method)
        method_rank.sort(key=operator.attrgetter('score'))
        fault_method = 0
        for i in range(len(method_rank)):
            if method_rank[i].fault == True:
                fault_method = i
        total.append(fault_method)
    return numpy.mean(total),numpy.std(total)


def train(dataset):
    toolbox.register("evaluate", evalFitness, dataset=dataset)

    pop = toolbox.population(n=40)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("min", numpy.min, axis=0)

    algorithms.eaSimple(pop, toolbox, 1.0, 0.1, ngen=100, stats=mstats, halloffame=hof, verbose=True)

    return hof[0]
