from torch.utils.data import Dataset 
import bisect 


class PhoConcatDataset(Dataset):
    """
    Meta-style composed dataset.
    Routes (idx, image_num, aspect_ratio) to the correct sub-dataset.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = self._compute_cumulative_sizes()

    def _compute_cumulative_sizes(self):
        sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            sizes.append(total)
        return sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _get_dataset_index(self, global_idx):
        """
        Convert global idx -> (dataset_id, local_idx)
        """
        dataset_id = bisect.bisect_right(self.cumulative_sizes, global_idx)
        prev_size = 0 if dataset_id == 0 else self.cumulative_sizes[dataset_id - 1]
        local_idx = global_idx - prev_size
        return dataset_id, local_idx


    def __getitem__(self, items):
        """
        items = (global_idx, image_num, aspect_ratio)
        """
        
        global_idx, image_num, aspect_ratio = items
        print('concat dataset: ', global_idx, image_num, aspect_ratio)

        dataset_id, local_idx = self._get_dataset_index(global_idx)

        return self.datasets[dataset_id][
            (local_idx, image_num, aspect_ratio)
        ]