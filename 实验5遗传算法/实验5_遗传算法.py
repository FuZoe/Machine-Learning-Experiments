import numpy as np
import matplotlib.pyplot as plt


# 定义目标函数
def target_function(x):
    return x * np.sin(10 * np.pi * x) + 2


# 生成数据范围
x_values = np.linspace(-1, 2, 100)

np.random.seed(42)  # 设置随机数种子，确保生成的随机数一致

# 创建初始种群
initial_population = np.random.uniform(-1, 2, 10)  # 在范围 [-1, 2) 内生成 10 个随机浮点数

# 输出初始种群及其适应度
for individual, fitness in zip(initial_population, target_function(initial_population)):
    print(f"Individual={individual:.2f}, Fitness={fitness:.2f}")

# 绘制目标函数及初始种群点
plt.plot(x_values, target_function(x_values), label="Target Function")
plt.scatter(initial_population, target_function(initial_population), color='red', label="Initial Population")
plt.legend()
plt.show()


# 编码函数：将浮点数种群转为二进制编码
def binary_encode(data, min_val=-1, max_val=2, resolution=2**18):
    normalized = (data - min_val) / (max_val - min_val) * resolution
    return np.array([np.binary_repr(int(val), width=18) for val in normalized])


# 解码函数：将二进制数据还原为浮点数
def binary_decode(binary_data, min_val=-1, max_val=2, resolution=2**18):
    return np.array([(int(bits, 2) / resolution) * (max_val - min_val) + min_val for bits in binary_data])


# 编码初始种群
encoded_population = binary_encode(initial_population)

# 输出编码后的种群信息
for individual, binary, fitness in zip(initial_population, encoded_population, target_function(initial_population)):
    print(f"Individual={individual:.2f}, Binary={binary}, Fitness={fitness:.2f}")

# 解码并计算适应度
decoded_population = binary_decode(encoded_population)
fitness_values = target_function(decoded_population)

for binary, decoded, fitness in zip(encoded_population, decoded_population, fitness_values):
    print(f"Binary={binary}, Decoded={decoded:.2f}, Fitness={fitness:.2f}")

# 将适应度正则化
fitness_values = fitness_values - fitness_values.min() + 1e-6


# 选择和交叉函数
def select_and_crossover(chromosomes, fitness_scores, crossover_rate=0.6):
    selection_prob = fitness_scores / fitness_scores.sum()
    cumulative_prob = np.cumsum(selection_prob)
    random_selections = np.random.rand(len(fitness_scores))
    selected_chromosomes = np.array([chromosomes[np.searchsorted(cumulative_prob, rnd)] for rnd in random_selections])

    # 随机配对进行交叉
    num_crossovers = int(len(selected_chromosomes) * crossover_rate // 2 * 2)
    pairs = np.random.permutation(num_crossovers).reshape(-1, 2)
    midpoint = len(selected_chromosomes[0]) // 2
    for i, j in pairs:
        selected_chromosomes[i], selected_chromosomes[j] = (
            selected_chromosomes[i][:midpoint] + selected_chromosomes[j][midpoint:],
            selected_chromosomes[j][:midpoint] + selected_chromosomes[i][midpoint:]
        )
    return selected_chromosomes


# 变异函数
def mutate_population(chromosomes, mutation_rate=0.1):
    mutated_chromosomes = []
    for chrom in chromosomes:
        if np.random.rand() < mutation_rate:
            mutation_pos = np.random.randint(len(chrom))
            chrom = chrom[:mutation_pos] + ('1' if chrom[mutation_pos] == '0' else '0') + chrom[mutation_pos + 1:]
        mutated_chromosomes.append(chrom)
    return np.array(mutated_chromosomes)


# 绘图函数：比较两代种群
def compare_generations(old_population, new_population, fitness_func):
    x_vals = np.linspace(-1, 2, 100)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, population, title in zip(axes, [old_population, new_population], ["Old Generation", "New Generation"]):
        decoded = binary_decode(population)
        fitness_scores = fitness_func(decoded)
        ax.plot(x_vals, fitness_func(x_vals), label="Fitness Curve")
        ax.scatter(decoded, fitness_scores, color='red', label="Population")
        ax.legend()
        ax.set_title(title)

    plt.show()


# 初始化并运行迭代过程
np.random.seed(42)
population = np.random.uniform(-1, 2, 100)
encoded_population = binary_encode(population)

for generation in range(1000):
    fitness_values = target_function(binary_decode(encoded_population))
    fitness_values = fitness_values - fitness_values.min() + 1e-6

    next_generation = mutate_population(select_and_crossover(encoded_population, fitness_values))
    if generation % 300 == 1:
        compare_generations(encoded_population, next_generation, target_function)
    encoded_population = next_generation

compare_generations(encoded_population, mutate_population(encoded_population), target_function)
