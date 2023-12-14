# Random polymers processing

This project contains the functions I have used over the last few months for processing the random polymers data. It includes optimized model parameters (using ax optimization), the trained models for both absolute and relative models, and the data files that I import in the code. 

## Configuration

I use my anaconda3/base/bin environment. For some reason, the package versions on it just work when they don't work for other environments. 

## File Descriptions

The best_parameters* files are the optimized parameters for the models. 
The embedding_model_noavg_* files are the saved trained models. 
The embedding_processing.ipynb file is the main file that has the code I use to process my data. 
The no_avg_dataset.csv file is the UMAP embedding file, where the runs are not averaged. So, a normal UMAP output file. 
The random_polymers_dataset_manipulation.py file is past code used to process data at the very beginning of this project. 

## Organization

I have a few cells that are designated as "always run." These have to be run even if you aren't training a model, and are needed for when you load a pre-trained model. I separated my training into my absolute position model (output Z0, Z1 coordinates at a given p value) and my relative position model (output vector from initial Z0, Z1 coordinates to Z0, Z1 coordinates at a given p value). I have KDE code and some norm calculation code after model training.

## Other notes

I have a few places where I intentionally ignored errors in my embedding code. You can see one example in the second cell, where I leave out a set of replicas from the dataset if there aren't five replicas. I need to fix this, I just completely forgot that I did that until recently and haven't had a chance to figure out how to change that. It is only a few datapoints out of 13,000+ though, so I don't think it is the most pressing issue. 
