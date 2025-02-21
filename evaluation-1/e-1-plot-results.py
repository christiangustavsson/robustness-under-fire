import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == "__main__":

    # Creating target or goal for the excersise
    target_model_RMSE = np.ones(100, dtype=float) * 0.4153667656904239

    # Loading the common x-axis, 1-100
    index = np.arange(1,101).reshape(100,)

    # Loading the datasets 
    with open('evaluation-1/results/e-1-RR/tracker_effectiveness_average.pickle', 'rb') as file:
        tracker_e1RR_efficiency_average = pickle.load(file)
    
    with open('evaluation-1/results/e-1-RR/tracker_effectiveness_all.pickle', 'rb') as file:
        tracker_e1RR_efficiency_all = pickle.load(file)
    tracker_best_e1RR = np.min(tracker_e1RR_efficiency_all, axis=0)  

    with open('evaluation-1/results/e-1-SVR/tracker_effectiveness_average.pickle', 'rb') as file:
        tracker_e1SVR_efficiency_average = pickle.load(file)
    
    with open('evaluation-1/results/e-1-SVR/tracker_effectiveness_all.pickle', 'rb') as file:
        tracker_e1SVR_efficiency_all = pickle.load(file)     
    tracker_best_e1SVR = np.min(tracker_e1SVR_efficiency_all, axis=0)   

    # Loading e-1-GPR datasets   
    with open('evaluation-1/results/e-1-GPR/tracker_effectiveness_average.pickle', 'rb') as file:
        tracker_e1GPR_efficiency_average = pickle.load(file)

    with open('evaluation-1/results/e-1-GPR/tracker_effectiveness_all.pickle', 'rb') as file:
        tracker_e1GPR_efficiency_all = pickle.load(file)
    tracker_best_e1GPR = np.min(tracker_e1GPR_efficiency_all, axis=0)

    tracker_best_all = [min(a, b, c) for a, b, c in zip(tracker_best_e1SVR, tracker_best_e1RR, tracker_best_e1GPR)]

    # Set the Seaborn style to "white" (no background color)
    sns.set(style="white")

    # Increase font size by adjusting rcParams for various plot elements
    plt.rcParams.update({
        'axes.labelsize': 18,    # Axis labels font size
        'axes.titlesize': 18,    # Title font size
        'xtick.labelsize': 18,   # X-axis tick labels font size
        'ytick.labelsize': 18,   # Y-axis tick labels font size
        'legend.fontsize': 18,   # Legend font size
        'figure.titlesize': 22,  # Figure title font size
    })
    
    # Define colors and font settings
    blue = "#003366"
    light_blue = sns.color_palette("Blues")[2]  # Light blue color from the Blues palette
    reddish = sns.color_palette("Reds")[3]  # A moderate red color from the Reds palette
    light_reddish = sns.light_palette(reddish, reverse=True, as_cmap=False)[2]  # Lighter version of the red
    orange = sns.color_palette("Oranges")[3]  # A nice moderate orange from the Oranges palette
    light_orange = sns.light_palette(orange, reverse=True, as_cmap=False)[2]  # Lighter version of the orange
    green = "#2ca02c" # Darker green
    light_green = sns.light_palette(green, n_colors=1)[0]  # Lighter green
    black = "black"

    # A set of plots will be created

    # Create the plot
    plt.figure(figsize=(10, 8))

    for e in range(10000):
        plt.plot(index, tracker_e1RR_efficiency_all[e], color=light_blue, alpha=0.01)  

    for e in range(10000):
        plt.plot(index, tracker_e1SVR_efficiency_all[e], color=light_green, alpha=0.01)

    for e in range(10000):
        plt.plot(index, tracker_e1GPR_efficiency_all[e], color=light_reddish, alpha=0.01)

    plt.plot(index, tracker_e1RR_efficiency_average.reshape(100,), label=r"$ \mu_{s_1} $ (RR) average loss", color=blue, linewidth=3)      

    plt.plot(index, tracker_e1SVR_efficiency_average.reshape(100,), label=r"$ \mu_{s_2} $ (SVR) average loss", color=green, linewidth=3)        

    plt.plot(index, tracker_e1GPR_efficiency_average.reshape(100,), label=r"$ \mu_{s_3} $ (GPR) average loss", color=reddish, linewidth=3)

    plt.plot(index, target_model_RMSE, label=r"$ \mu_t $ (RR)", linestyle='--', color=black, linewidth=3)
 

    title_font = {'size':'16', 'color':'black', 'weight':'normal',
                'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'size':'14'}

    # Set logarithmic scale for the y-axis
    plt.yscale("log")

    # Axis labels and title
    plt.xlabel("Queries", **axis_font)
    plt.ylabel("RMSE loss", **axis_font)
    plt.title(r"$ S $ prediction RMSE loss, trained on 1 - 100 queries", **title_font)

    # Show legend and plot
    plt.legend()
    plt.show()    


    # PLOT 2
    plt.figure(figsize=(10, 8))

    plt.plot(index, tracker_best_e1RR.reshape(100,), label=r"$ \mu_{s_1} $ (RR) min loss", color=blue, linestyle=':', linewidth=3) 

    plt.plot(index, tracker_best_e1SVR.reshape(100,), label=r"$ \mu_{s_2} $ (SVR) min loss", color=green, linestyle=':', linewidth=3)

    plt.plot(index, tracker_best_e1GPR.reshape(100,), label=r"$ \mu_{s_3} $ (GPR) min loss", color=reddish, linestyle=':', linewidth=3)

    plt.plot(index, target_model_RMSE, label=r"$ \mu_t $ (RR)", linestyle='--', color=black, linewidth=3)
 

    title_font = {'size':'16', 'color':'black', 'weight':'normal',
                'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'size':'14'}

    # Set logarithmic scale for the y-axis
    plt.yscale("log")

    # Axis labels and title
    plt.xlabel("Queries", **axis_font)
    plt.ylabel("RMSE loss", **axis_font)
    # plt.title(r"$ S $ prediction RMSE loss, trained on 1 - 100 queries", **title_font)
    plt.title(r"$ S $ prediction min RMSE loss, trained on 1 - 100 queries", **title_font)

    # Show legend and plot
    plt.legend()
    plt.show()    


    # PLOT 3
    plt.figure(figsize=(10, 8))

    plt.plot(index, target_model_RMSE, label=r"$ \mu_t $ (RR)", linestyle='--', color=black, linewidth=3)
    plt.plot(index, tracker_best_all, label=r"$ \mu_{s-best} $ at each step", color=black, linestyle=':', linewidth=3)

    title_font = {'size':'16', 'color':'black', 'weight':'normal',
                'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'size':'14'}

    # Set logarithmic scale for the y-axis
    plt.yscale("log")

    # Axis labels and title
    plt.xlabel("Queries", **axis_font)
    plt.ylabel("RMSE loss", **axis_font)
    plt.title(r"Best performing $ \mu_{s_j} $ at each step, 1 - 100 queries", **title_font)

    # Show legend and plot
    plt.legend()
    plt.show()    
    
