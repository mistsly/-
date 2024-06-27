import random
from full_net import Net
import torch
import pandas as pd


# 定义个体类，代表种群中的一个个体
class Individual:
    def __init__(self, genes):
        self.genes = genes  # 个体的基因序列
        self.genes_trans = self.get_trans()
        self.out_prd = 0
        self.fitness = self.calculate_fitness()  # 个体的适应度

    def calculate_fitness(self):
        """
        得分离1越近越好，得分越高
        :return: 得分
        """
        with torch.no_grad():
            X_test = torch.tensor(self.genes_trans).float()
            X_test = X_test.to(device)
            output = net(X_test)
            self.out_prd = output.item()
        return 1 / abs(1.0 - self.out_prd)

    def get_trans(self):
        """
        个体基因生成是根据最大最小值定的，但神经网络输入是经过标准化的，这一步就是标准化
        :return: 基因标准化结果
        """
        x_list = []
        for x, mean, Var in zip(self.genes, mean_list, var_list):
            x_list.append(((x-mean)/Var))
        return x_list


def get_genes():
    gene = []
    # 给出个体基因
    for M, m in zip(max_list, min_list):
        gene.append(random.uniform(m, M))
    return gene


# 初始化种群
def initialize_population(size, gene_length):
    # size: 种群的大小
    # gene_length: 个体基因序列的长度
    # 生成初始种群，每个个体由随机生成的基因序列组成
    populations = []
    for _ in range(size-10):
        populations.append(Individual(get_genes()))
    for _ in range(10):
        gene_i_list = []
        for gene_i, mean_i in zip(input_gene, mean_list):
            gene_i_list.append(gene_i+random.uniform(-mean_i,mean_i))
        populations.append(Individual(gene_i_list))
    return populations


# 选择过程
def selection(population, num_parents):
    # 根据适应度排序，选择适应度最高的个体作为父母
    # population: 当前种群
    # num_parents: 选择的父母数量
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    return sorted_population[:num_parents]


# 交叉过程
def crossover(parent1, parent2):
    # 单点交叉
    # parent1, parent2: 选择的两个父本个体
    # 随机选择交叉点，交换父本基因，生成两个子代
    point = random.randint(1, len(parent1.genes) - 1)
    child1_genes = parent1.genes[:point] + parent2.genes[point:]
    child2_genes = parent2.genes[:point] + parent1.genes[point:]
    return Individual(child1_genes), Individual(child2_genes)


# 变异过程
def mutation(individual, mutation_rate=0.1):
    # 对个体的基因序列进行随机变异
    # individual: 要变异的个体
    # mutation_rate: 变异概率
    for i in range(len(individual.genes)):
        if random.random() < mutation_rate:
            # 对每个基因位以一定的概率进行增减操作
            if individual.genes[i] > 2:
                individual.genes[i] += random.uniform(-2, 2)
            else:
                individual.genes[i] += random.uniform(-individual.genes[i], 2)
    # 更新个体的适应度
    individual.fitness = individual.calculate_fitness()


# 遗传算法主函数
def genetic_algorithm(population_size, gene_length, num_generations):
    # population_size: 种群大小
    # gene_length: 基因长度
    # num_generations: 进化代数
    # 初始化种群
    population = initialize_population(population_size, gene_length)
    for _ in range(num_generations):
        # 选择
        parents = selection(population, population_size // 2)
        next_generation = []
        # 生成新一代
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            mutation(child1)
            mutation(child2)
            next_generation.extend([child1, child2])
        population = next_generation
        # 每一代选出适应度最高的个体
        best_individual = max(population, key=lambda x: x.fitness)
        print(f"最优适应度: {best_individual.fitness}")
        print(f"此时的产量：{best_individual.out_prd}")
    return best_individual


def main_gene(opt_s='L'):
    global opt
    global net
    global device
    global max_list
    global min_list
    global mean_list
    global var_list
    global input_gene
    opt = opt_s
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(opt).to(device)
    net.load_state_dict(torch.load(f'{opt}_model_params.pth'))
    params = pd.read_csv(f'./{opt}_combined_values.csv')
    print(params)
    max_list = params.iloc[0, :]
    min_list = params.iloc[1, :]
    mean_list = params.iloc[2, :]
    var_list = params.iloc[3, :]
    input_gene = params.iloc[4, :]
    # todo
    # input_gene这个地方是需要从前端获取的
    best = genetic_algorithm(200, len(max_list), 70 + 10)
    print(f"原始输入参数：{input_gene}")
    print(f"最优个体基因: {best.genes}")
    print(f"建议修改意见：{best.genes-input_gene}")
#     todo
# 建议修改意见也需要传到前端去


if __name__ == '__main__':
    main_gene('L')
