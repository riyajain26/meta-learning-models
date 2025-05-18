import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EEGMetaDataset(Dataset):
    """
    Meta-learning dataset for EEG data, designed for MAML.
    
    Each sample (task) consists of:
    - A support set (used for adaptation)
    - A query set (used for meta-update)
    
    Assumes EEG features are stored in .npy files, and task information is in a DataFrame.
    """

    def __init__(self, df, data_dir, num_tasks, k_shot=1, q_query=1):
        """
        Args:
            df (pd.DataFrame): DataFrame with columns ['file', 'label', 'participant']
            data_dir (str): Directory containing EEG .npy files
            num_tasks (int): Number of meta-learning tasks per epoch
            k_shot (int): Number of examples per class in the support set
            q_query (int): Number of examples per class in the query set
        """
        self.df = df
        self.data_dir = data_dir
        self.num_tasks = num_tasks
        self.k_shot = k_shot
        self.q_query = q_query
         
        self.num_classes = len(self.df['label'].unique())
        self.participants = df['participant'].unique()

        self.tasks = self.create_tasks()    ## Create meta-learning tasks

    def create_tasks(self):
        """
        Generates meta-learning tasks by randomly sampling participants.
        Each task includes k-shot support and q-query query examples for each class.
        """
        tasks = []

        for _ in range(self.num_tasks):
            ## Randomly choose one participant per task 
            participant = random.choice(self.participants)
            p_data = self.df[self.df['participant'] == participant]

            support_set, query_set = [], []

            ## Create support and query sets for each task (ensuring each label is there in each set) 
            for label in p_data['label'].unique():
                samples = p_data[p_data['label']==label]

                # Decide whether to sample with or without replacement
                sample_with_replacement = len(samples) < (self.k_shot + self.q_query)
                chosen = samples.sample(
                    n=self.k_shot + self.q_query,
                    replace=sample_with_replacement
                )

                support_set.append(chosen.iloc[:self.k_shot])
                query_set.append(chosen.iloc[self.k_shot:])

            support_df = pd.concat(support_set)
            query_df = pd.concat(query_set)
            tasks.append((support_df, query_df))
            
        return tasks

    def __len__(self):
        return len(self.tasks)   

    def __getitem__(self, idx):
        """
        Returns:
            support_x (Tensor): EEG signals in support set, shape [N_support, C, T]
            support_y (Tensor): Labels for support set
            query_x (Tensor): EEG signals in query set, shape [N_query, C, T]
            query_y (Tensor): Labels for query set
        """
        support_df, query_df = self.tasks[idx]

        ## Get EEG signals (.npy) and labels extracted from dataFrame (.csv file)
        def load_data(df):
            signals, labels = [], []
            for _, row in df.iterrows():
                file_path = os.path.join(self.data_dir, row['file'])
                try: 
                    x = np.load(file_path) 
                    x = torch.tensor(x, dtype=torch.float32)
                    y = int(row['label'])
                    signals.append(x)
                    labels.append(y)
                except Exception as e:
                    print(f"[Warning] Failed to load file {file_path}: {e}")
                    continue
                
            return torch.stack(signals), torch.tensor(labels)

        support_x, support_y = load_data(support_df)
        query_x, query_y = load_data(query_df)

        return support_x, support_y, query_x, query_y