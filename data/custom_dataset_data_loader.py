import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'labeled':
        from data.labeled_dataset import LabeledDataset
        dataset = LabeledDataset()
    elif opt.dataset_mode == 'unlabeled':
        from data.unlabeled_dataset import UnlabeledDataset
        dataset = UnlabeledDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def get_dataframe(self):
        return self.dataset.get_dataset_labels_df()

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
