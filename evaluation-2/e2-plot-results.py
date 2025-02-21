import matplotlib.pyplot as plt
import seaborn as sns

# The results below are added from the material in evaluation-2/results

xaxis = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 
         14000, 15000, 16000, 17000, 18000, 19000, 20000]

mut = [0.8133, 0.8133, 0.8133, 0.8133, 0.8133, 0.8133, 0.8133, 0.8133, 0.8133, 0.8133,
      0.8133, 0.8133, 0.8133, 0.8133, 0.8133, 0.8133, 0.8133, 0.8133, 0.8133, 0.8133]
mut = [x * 100 for x in mut]

# ResNet-18
mu1 = [0.4106, 0.4994, 0.5157, 0.5385, 0.5786, 0.5815, 0.5972, 0.6181, 0.6339, 0.6326, 
       0.6357, 0.6457, 0.6639, 0.6563, 0.6589, 0.6493, 0.6809, 0.6808, 0.6776, 0.6798]
mu1 = [x * 100 for x in mu1]

# ResNet-152
mu2 = [0.3250, 0.3997, 0.4768, 0.5048, 0.5461, 0.5754, 0.5833, 0.6121, 0.6151, 0.6067, 
       0.6341, 0.6410, 0.6407, 0.6578, 0.6651, 0.6683, 0.6645, 0.6799, 0.6809, 0.6687]
mu2 = [x * 100 for x in mu2]

# VGG-11
mu3 = [0.4342, 0.5134, 0.5203, 0.5433, 0.5912, 0.6157, 0.6146, 0.6419, 0.6463, 0.6705,
       0.6786, 0.6799, 0.6841, 0.6770, 0.6744, 0.6851, 0.6956, 0.7097, 0.7135, 0.7131]
mu3 = [x * 100 for x in mu3]

# VGG-19
mu4 = [0.3828, 0.4401, 0.5178, 0.4569, 0.5844, 0.6054, 0.5292, 0.6097, 0.6471, 0.6570,
       0.6632, 0.6648, 0.6878, 0.6952, 0.6827, 0.7069, 0.7068, 0.7127, 0.7208, 0.6940]
mu4 = [x * 100 for x in mu4]

# DenseNet-121
mu5 = [0.4569, 0.5184, 0.5651, 0.5743, 0.6159, 0.6177, 0.6331, 0.6469, 0.6447, 0.6659,
       0.6702, 0.6754, 0.6869, 0.6927, 0.6857, 0.6995, 0.6956, 0.7025, 0.7113, 0.7181]
mu5 = [x * 100 for x in mu5]

# DenseNet-161
mu6 = [0.4398, 0.5003, 0.5483, 0.5982, 0.6183, 0.6209, 0.6461, 0.6584, 0.6663, 0.6735,
       0.6792, 0.6837, 0.6979, 0.6990, 0.7026, 0.7033, 0.7143, 0.7147, 0.7120, 0.7216]
mu6 = [x * 100 for x in mu6]

# Upper bound
muu = [0.4569, 0.5184, 0.5651, 0.5982, 0.6183, 0.6209, 0.6461, 0.6584, 0.6663, 0.6735,
       0.6792, 0.6837, 0.6979, 0.6990, 0.7026, 0.7069, 0.7143, 0.7147, 0.7208, 0.7216]
muu = [x * 100 for x in muu]


# This code results in two plots.

sns.set(style="whitegrid")

# Increase font size by adjusting rcParams for various plot elements
plt.rcParams.update({
    'axes.labelsize': 18,    # Axis labels font size
    'axes.titlesize': 18,    # Title font size
    'xtick.labelsize': 12,   # X-axis tick labels font size
    'ytick.labelsize': 12,   # Y-axis tick labels font size
    'legend.fontsize': 18,   # Legend font size
    'figure.titlesize': 22,  # Figure title font size
})

black = "black"

# PLOT 1

plt.figure(figsize=(20, 8))
sns.lineplot(x=xaxis, y=mu1, label=r'$ \mu_{s_1} $(ResNet-18)', linewidth=3)
sns.lineplot(x=xaxis, y=mu2, label=r'$ \mu_{s_2} $(ResNet-152)', linewidth=3)
sns.lineplot(x=xaxis, y=mu3, label=r'$ \mu_{s_3} $ (VGG-11)', linewidth=3)
sns.lineplot(x=xaxis, y=mu4, label=r'$ \mu_{s_4} $ (VGG-19)', linewidth=3)
sns.lineplot(x=xaxis, y=mu5, label=r'$ \mu_{s_5} $ (DenseNet-121)', linewidth=3)
sns.lineplot(x=xaxis, y=mu6, label=r'$ \mu_{s_6} $ (DenseNet-161)', linewidth=3)
sns.lineplot(x=xaxis, y=mut, label=r'$ \mu_{t} $ (ResNet-18)', linestyle='--', color=black, linewidth=3)
# sns.lineplot(x=xaxis, y=muu, label='Theoretical upper bound', linestyle=':', color=black, linewidth=3)


plt.title(r'$ S $ prediction accuarcy, trained on 1,000 - 20,0000 queries')
plt.xlabel('Queries')
plt.ylabel('Accuracy (%)')
plt.xticks(ticks=range(1000, 21000, 1000))
plt.yticks(ticks=range(0, 110, 20))
plt.legend()
plt.show()


# PLOT 2

plt.figure(figsize=(10, 8))
sns.lineplot(x=xaxis, y=mut, label=r'$ \mu_{t} $ (ResNet-18)', linestyle='--', color=black, linewidth=3)
sns.lineplot(x=xaxis, y=muu, label=r'$ \mu_{s-best}$ at each step', linestyle=':', color=black, linewidth=3)

plt.title(r'Best performing $\mu_{s_j}$ at each step, 1,000 - 20,0000 queries')
plt.xlabel('Queries')
plt.ylabel('Accuracy (%)')
plt.xticks(ticks=range(2000, 21000, 2000))
plt.yticks(ticks=range(0, 110, 20))
plt.legend()
plt.show()