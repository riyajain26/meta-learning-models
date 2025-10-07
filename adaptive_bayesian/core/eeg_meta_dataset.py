import torch
from torch.utils.data import Dataset, DataLoader


class EEGDataset(Dataset):
    """
    Dataset class for EEG data
    Expects data organized by subjects
    """
    def __init__(self, data, labels, subjects):
        """
        Args:
            data: (N, n_channels, time_length) - all EEG samples
            labels: (N,) - class labels
            subjects: (N,) - subject IDs
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.subjects = torch.LongTensor(subjects)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.subjects[idx]
    
    def get_subject_data(self, subject_id):
        """Get all data for a specific subject"""
        mask = self.subjects == subject_id
        return self.data[mask], self.labels[mask]
    
    def get_subjects_data(self, subject_ids):
        """Get all data for multiple subjects"""
        mask = torch.zeros(len(self.subjects), dtype=torch.bool)
        for sid in subject_ids:
            mask |= (self.subjects == sid)
        return self.data[mask], self.labels[mask]

