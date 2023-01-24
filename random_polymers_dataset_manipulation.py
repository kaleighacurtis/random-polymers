#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 19:19:15 2023

@author: kaleighcurtis
"""
#first I need to import the csv file 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

#Getting the dataset, sometimes this doesn't work. Need to be in downloads folder.
dataset = pd.read_csv("/Users/kaleighcurtis/Downloads/embeddings copy.csv")
dataset_alpha = dataset

#
#%% Defining function to extract p from file name

def extract_p(file_name):
    sep_file_name = file_name.split("_")
    p_with_junk = sep_file_name[-1]
    no_gsd = p_with_junk.replace(".gsd",'')
    p_val_string = no_gsd.replace("p",'')

    p_val = float(p_val_string)
    
    return p_val

#%% Collected all p values and added that to the dataframe

p_vals = []
for file in dataset["Filename"]:
    p = extract_p(file)
    p_vals.append(p)

dataset['p'] = p_vals

p_vals_a =[]
for file in dataset_alpha["Filename"]:
    p = extract_p(file)
    p_vals_a.append(p)
dataset_alpha['p']= p_vals_a

#%% Getting information for the plotting function
 
#ONLY LOOKING AT FRAME 4
sequences = dataset["Sequence"].unique()
dataset = dataset[dataset["Frame"] ==4]
df1 = dataset[dataset["Sequence"] == sequences[0]]
df2 = dataset[dataset["Sequence"] == sequences[1]]
df3 = dataset[dataset["Sequence"] == sequences[2]]
df4 = dataset[dataset["Sequence"] == sequences[3]]
df5 = dataset[dataset["Sequence"] == sequences[4]]

split_dataset_list = [df1, df2, df3, df4, df5]

df1 = dataset_alpha[dataset_alpha["Sequence"] == sequences[0]]
df2 = dataset_alpha[dataset_alpha["Sequence"] == sequences[1]]
df3 = dataset_alpha[dataset_alpha["Sequence"] == sequences[2]]
df4 = dataset_alpha[dataset_alpha["Sequence"] == sequences[3]]
df5 = dataset_alpha[dataset_alpha["Sequence"] == sequences[4]]

split_dataset_alpha = [df1, df2, df3, df4, df5]


seq_names = ["Micelles", "Worm-like micelles", "Structured Liquid", "Strings", "Membranes"]

#%% Function from github that will make the alphashape

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges



#%% Making a plot that will make the subplot I want
def plot_p_drift(split_dataset_list, seq_names, alpha, alpha_dataset):
    
    fig, axs = plt.subplots(5, figsize = (7,15), sharey= True)
    
    z0s = np.asarray(alpha_dataset["Z0"]).reshape(-1,1)
    z1s = np.asarray(alpha_dataset["Z1"]).reshape(-1,1)

    z_array = np.concatenate((z0s, z1s), axis = 1)
    
    z0s_alpha = np.asarray(alpha_dataset["Z0"]).reshape(-1,1)
    z1s_alpha = np.asarray(alpha_dataset["Z1"]).reshape(-1,1)
    zalpha_array = np.concatenate((z0s_alpha, z1s_alpha), axis = 1)

    for seq in range(5):
        df = split_dataset_list[seq]
        print("Running", seq_names[seq])
        
        ps = df["p"].unique()

        df_0_01 = df[df["p"]== ps[0]]
        df_0_05 = df[df["p"]== ps[1]]
        df_0_07 = df[df["p"]== ps[2]]
        df_0_15 = df[df["p"]== ps[3]]
        df_0_1  = df[df["p"]== ps[4]]
        
        
        df_0 = df[df["p"] == ps[5]]
        df0_avg = df_0.mean(axis = 0)
        z0_avg = df0_avg["Z0"]
        z1_avg = df0_avg["Z1"]
    
        #Plotting
        ax = axs[seq]
        ax.set_title(seq_names[seq])

        
        al_shape = alpha_shape(zalpha_array, alpha)

        for i, j in al_shape:
            ax.plot(zalpha_array[[i, j], 0], zalpha_array[[i, j], 1], color = 'black')

       


        ax.plot(z0_avg, z1_avg, marker = 'X', markersize = 15, color = 'black')
        
        ax.scatter(df_0_01["Z0"], df_0_01["Z1"], color = 'red', label = '0.01')
        ax.scatter(df_0_05["Z0"], df_0_05["Z1"], color = 'darkorange', label = "0.05")
        ax.scatter(df_0_07["Z0"], df_0_07["Z1"], color = 'forestgreen', label = "0.07")
        ax.scatter(df_0_15["Z0"], df_0_15["Z1"], color = 'purple', label = '0.15')
        ax.scatter(df_0_1["Z0"], df_0_1["Z1"], color = 'royalblue', label = '0.1')
        ax.set(xlabel = "Z0", ylabel = "Z1")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'right')
    fig.tight_layout()
        
    return fig.show()

#%%
plot_p_drift(split_dataset_list, seq_names, alpha = 0.97, alpha_dataset = dataset_alpha)


#%% wrapper function

def wrap_plot_p_drift(dataset, alpha):
    p_vals = []
    for file in dataset["Filename"]:
        p = extract_p(file)
        p_vals.append(p)
    dataset['p'] = p_vals
    
    sequences = dataset["Sequence"].unique()
    df1 = dataset[dataset["Sequence"] == sequences[0]]
    df2 = dataset[dataset["Sequence"] == sequences[1]]
    df3 = dataset[dataset["Sequence"] == sequences[2]]
    df4 = dataset[dataset["Sequence"] == sequences[3]]
    df5 = dataset[dataset["Sequence"] == sequences[4]]

    split_dataset_list = [df1, df2, df3, df4, df5]


    seq_names = ["Micelles", "Worm-like micelles", "Structured Liquid", "Strings", "Membranes"]
    
    plot_p_drift(split_dataset_list, seq_names, alpha)
    
    
wrap_plot_p_drift(dataset, alpha = 0.97)



#%% Next up is a line plot of p vs variance

#Some necessary functions
def get_avg(df, ps):
    avg_list = []
    for p in range(6):
            
        averaging_data = df[df["p"]==ps[p]]
        average_columns = averaging_data.mean(axis = 0)
        z0_avg = average_columns["Z0"]
        z1_avg = average_columns["Z1"]
        avg_list.append([z0_avg, z1_avg])
    return avg_list
    
def euclid(point1, point2):
    diff_x = (point2[0] - point1[0])**2
    diff_y = (point2[1] - point1[1])**2
        
    radicand = diff_x + diff_y
    d = np.sqrt(radicand)
        
    return d

#Getting z
z0s = np.asarray(dataset["Z0"]).reshape(-1,1)
z1s = np.asarray(dataset["Z1"]).reshape(-1,1)
z_array = np.concatenate((z0s, z1s), axis = 1)


def p_vs_variance(split_dataset_list, seq_names):
    
    #Need to get list of colors and marker shapes
    colors = ['red', 'darkorange', 'forestgreen', 'purple', 'royalblue']
    markers = ['d', "X", 'v', '*', 'o']
    
    for seq in range(5):
        df = split_dataset_list[seq]
        print("Running", seq_names[seq])
        
        ps = df["p"].unique()
        ps = sorted(list(ps))

        df_0    = df[df["p"]== ps[0]] 
        df_0_01 = df[df["p"]== ps[1]]
        df_0_05 = df[df["p"]== ps[2]]
        df_0_07 = df[df["p"]== ps[3]]
        df_0_1  = df[df["p"]== ps[4]]
        df_0_15 = df[df["p"]== ps[5]]
        
        
        df_lists = [df_0, df_0_01, df_0_05, df_0_07, df_0_1, df_0_15]
        var_list = []
        
        #Getting variances
        var_list = []
        avg_list = get_avg(df, ps)
        
        
        
        for num, data_f in enumerate(df_lists):
            dist_list = []
            average_point = avg_list[num]
            
            z0s = np.asarray(data_f["Z0"]).reshape(-1,1)
            z1s = np.asarray(data_f["Z1"]).reshape(-1,1)
            z_arr = np.concatenate((z0s, z1s), axis = 1)
            
            for coord in z_arr:
                euclid_distance = euclid(average_point, coord)
                dist_list.append(euclid_distance)
            
            dist_var = np.var(dist_list)
            var_list.append(dist_var)
        
        
        plt.plot(ps, var_list, color = colors[seq], marker = markers[seq], label = seq_names[seq])
    plt.xlabel('p')
    plt.ylabel('Variance')
    plt.legend()
    plt.title('p vs. variance in euclidian distance')
    return plt.show()
            
        
p_vs_variance(split_dataset_list, seq_names)
            



#%% Next is plotting the variance on top of the averages plots. I'm just going to modify an existing function. 

'''This function used to plot circles of the variance around the mean value of each p value for frame 4. '''

def plot_p_avg_drift(split_dataset_list, seq_names, alpha, plot_var):
    
    fig, axs = plt.subplots(5, figsize = (7,15), sharey= True)
    
    z0s = np.asarray(dataset["Z0"]).reshape(-1,1)
    z1s = np.asarray(dataset["Z1"]).reshape(-1,1)

    z_array = np.concatenate((z0s, z1s), axis = 1)


    def get_avg(df, ps):
        avg_list = []
        for p in range(6):
            
            averaging_data = df[df["p"]==ps[p]]
            average_columns = averaging_data.mean(axis = 0)
            z0_avg = average_columns["Z0"]
            z1_avg = average_columns["Z1"]
            avg_list.append([z0_avg, z1_avg])
        return avg_list
    
    
    def euclid(point1, point2):
        diff_x = (point2[0] - point1[0])**2
        diff_y = (point2[1] - point1[1])**2
        
        radicand = diff_x + diff_y
        d = np.sqrt(radicand)
        
        return d
        
        
    for seq in range(5):
        df = split_dataset_list[seq]
        print("Running", seq_names[seq])
        print("\n")
        
        ps = df["p"].unique()
        ps = sorted(list(ps))
        
        avg_list = get_avg(df, ps)
        

        #Plotting information
        ax = axs[seq]
        ax.set_title(seq_names[seq])
        
        
        #Separates dataset based off p
        df_0_01 = df[df["p"]== ps[1]]
        df_0_05 = df[df["p"]== ps[2]]
        df_0_07 = df[df["p"]== ps[3]]
        df_0_1  = df[df["p"]== ps[4]]
        df_0_15 = df[df["p"]== ps[5]]
        df_lists = [df_0_01, df_0_05, df_0_07, df_0_1, df_0_15]
        
        var_list = []
        #Getting variances, specifically variance of euclidian distance
        for num, data_f in enumerate(df_lists):
            dist_list = []
            average_point = avg_list[num]
            
            z0s = np.asarray(data_f["Z0"]).reshape(-1,1)
            z1s = np.asarray(data_f["Z1"]).reshape(-1,1)
            z_arr = np.concatenate((z0s, z1s), axis = 1)
            
            for coord in z_arr:
                euclid_distance = euclid(average_point, coord)
                dist_list.append(euclid_distance)
            
            dist_var = np.std(dist_list)
            var_list.append(dist_var)

        #Getting alpha shape and plotting it
        al_shape = alpha_shape(z_array, alpha)
        for i, j in al_shape:
            ax.plot(z_array[[i, j], 0], z_array[[i, j], 1], color = 'black')
            
        #Plotting averages for each p   
        ax.scatter(avg_list[0][0], avg_list[0][1], color = 'black', label = '0', marker = 'X')
        ax.scatter(avg_list[1][0], avg_list[1][1], color = 'red', label = '0.01')
        ax.scatter(avg_list[2][0], avg_list[2][1], color = 'darkorange', label = "0.05")
        ax.scatter(avg_list[3][0], avg_list[3][1], color = 'forestgreen', label = "0.07")
        ax.scatter(avg_list[4][0], avg_list[4][1], color = 'royalblue', label = '0.1')
        ax.scatter(avg_list[5][0], avg_list[5][1], color = 'purple', label = '0.15')

        if plot_var == True:
            #Variance circles
            var_01 = plt.Circle((avg_list[1][0], avg_list[1][1]), radius = var_list[0], hatch = '/', edgecolor = 'red', facecolor = None, fill = False)
            var_05 = plt.Circle((avg_list[2][0], avg_list[2][1]), radius = var_list[1], hatch = '/', edgecolor = 'darkorange', facecolor = None, fill = False)
            var_07 = plt.Circle((avg_list[3][0], avg_list[3][1]), radius = var_list[2], hatch = '/', edgecolor = 'forestgreen', facecolor = None, fill = False)
            var_1  = plt.Circle((avg_list[4][0], avg_list[4][1]), radius = var_list[3], hatch = '/', edgecolor = 'royalblue', facecolor = None, fill = False)
            var_15 = plt.Circle((avg_list[5][0], avg_list[5][1]), radius = var_list[4], hatch = '/', edgecolor = 'purple', facecolor = None, fill = False)
        
            ax.add_artist(var_01)
            ax.add_artist(var_05)
            ax.add_artist(var_07)
            ax.add_artist(var_1)
            ax.add_artist(var_15)
        
        ax.set(xlabel = "Z0", ylabel = "Z1")
        
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'right')
    fig.tight_layout()
        
    return fig.show()



plot_p_avg_drift(split_dataset_list, seq_names, alpha = 0.97, plot_var = True)

#%% New section for new work! Adding the covariance ellipse code
'''This code lets you plot the average coordinate for each p value for each structure, only looking at frame 4. It also plots covariance ellipses around the mean points.'''
from matplotlib import patches

def add_cov_ellipse(this_d, ax, color, alpha=0.5, mag=5.99):    
    cov = np.cov(np.array(list(this_d[1])).squeeze().T)
    vals, vecs = np.linalg.eig(cov)
    mu = np.mean(this_d[1], axis=0)
    theta = np.arctan(vecs[1, np.argmax(vals)] / vecs[0, np.argmax(vals)]) * 180 / np.pi    
    
    ue = patches.Ellipse(mu, np.sqrt(mag*np.max(vals)), np.sqrt(mag*np.min(vals)),angle=theta, alpha=0.167, color=color)
    
    ax.add_patch(ue)
    return


def plot_p_drift_cov(split_dataset_list, seq_names, alpha, plot_var, alpha_dataset):
    
    fig, axs = plt.subplots(5, figsize = (7,15), sharey= True)
    
    z0s = np.asarray(dataset["Z0"]).reshape(-1,1)
    z1s = np.asarray(dataset["Z1"]).reshape(-1,1)

    
    z0s_alpha = np.asarray(alpha_dataset["Z0"]).reshape(-1,1)
    z1s_alpha = np.asarray(alpha_dataset["Z1"]).reshape(-1,1)
    zalpha_array = np.concatenate((z0s_alpha, z1s_alpha), axis = 1)

    #Gets
    def get_avg(df, ps):
        avg_list = []
        for p in range(6):
            
            averaging_data = df[df["p"]==ps[p]]
            average_columns = averaging_data.mean(axis = 0)
            z0_avg = average_columns["Z0"]
            z1_avg = average_columns["Z1"]
            avg_list.append([z0_avg, z1_avg])
        return avg_list
    

    #Running through each sequence
    for seq in range(5):
        df = split_dataset_list[seq]
        print("Running", seq_names[seq])
        print("\n")
        
        ps = df["p"].unique()
        ps = sorted(list(ps))
        
        avg_list = get_avg(df, ps)
        

        #Plotting information
        ax = axs[seq]
        ax.set_title(seq_names[seq])
        
        
        #Separates dataset based off p
        df_0_01 = df[df["p"]== ps[1]]
        df_0_05 = df[df["p"]== ps[2]]
        df_0_07 = df[df["p"]== ps[3]]
        df_0_1  = df[df["p"]== ps[4]]
        df_0_15 = df[df["p"]== ps[5]]
        df_lists = [df_0_01, df_0_05, df_0_07, df_0_1, df_0_15]
        
        big_z = []
        #Getting variances, specifically variance of euclidian distance
        for num, data_f in enumerate(df_lists):            
            z0s = np.asarray(data_f["Z0"]).reshape(-1,1)
            z1s = np.asarray(data_f["Z1"]).reshape(-1,1)
            z_arr = np.concatenate((z0s, z1s), axis = 1)
         
            big_z.append(z_arr)

        #Getting alpha shape and plotting it. Uses ALL frames to get a better shape. 
        al_shape = alpha_shape(zalpha_array, alpha)
        for i, j in al_shape:
            ax.plot(zalpha_array[[i, j], 0], zalpha_array[[i, j], 1], color = 'black')
            
        #Plotting averages for each p   
        ax.scatter(avg_list[0][0], avg_list[0][1], color = 'black', label = '0', marker = 'X')
        ax.scatter(avg_list[1][0], avg_list[1][1], color = 'red', label = '0.01')
        ax.scatter(avg_list[2][0], avg_list[2][1], color = 'darkorange', label = "0.05")
        ax.scatter(avg_list[3][0], avg_list[3][1], color = 'forestgreen', label = "0.07")
        ax.scatter(avg_list[4][0], avg_list[4][1], color = 'royalblue', label = '0.1')
        ax.scatter(avg_list[5][0], avg_list[5][1], color = 'purple', label = '0.15')



        z_dict = {0:big_z[0], 1:big_z[1], 2:big_z[2], 3:big_z[3], 4:big_z[4]}
        if plot_var == True:
            add_cov_ellipse(list(z_dict.items())[0], ax, color = 'red')
            add_cov_ellipse(list(z_dict.items())[1], ax, color = 'darkorange')
            add_cov_ellipse(list(z_dict.items())[2], ax, color = 'forestgreen')
            add_cov_ellipse(list(z_dict.items())[3], ax, color = 'royalblue')
            add_cov_ellipse(list(z_dict.items())[4], ax, color = 'purple')
            
        ax.set(xlabel = "Z0", ylabel = "Z1")
        
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc = 'center right')
 
        
    return fig.show()




plot_p_drift_cov(split_dataset_list, seq_names, alpha = 0.97, plot_var = True, alpha_dataset = dataset_alpha)












