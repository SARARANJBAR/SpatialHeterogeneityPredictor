import sys
import os.path
import random
import torch
from data.base_dataset import BaseDataset
import numpy as np
import SimpleITK as sitk
import torchio as tio
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import rotate

class LabeledDataset(BaseDataset):

    def __len__(self):
        return len(self.images)

    def name(self):
        return 'LabeledDataset'

    def initialize(self, opt):
        print('\n', self.name())
        self.images = []
        self.labels = []
        self.num_classes = opt.num_classes
        self.opt = opt
        self.phase = opt.phase
        self.target = opt.target + ' LI'
        self.roisize = opt.roisize
        self.input_nc = len(opt.input_seq)
        self.input_sequences = opt.input_seq

        self.dirpath = os.path.join(opt.dataroot, 'labeled_skullstripped/all')

        self.paths = self.sort_paths()
        self.master_df = None
        self.labels_csv = os.path.join(opt.dataroot, 'AllBxCoords_ResizedResampled_labeled.csv')

        self.make_dataset()
        print('\ndataset size:', len(self.images))

    def get_dataset_labels_df(self):
        return self.master_df

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in ['.nii', '.nii.gz'])

    def sort_paths(self):
        assert os.path.isdir(self.dirpath), '%s is not a valid directory' % self.dirpath

        # get all image paths
        paths = []
        for root, _, fnames in sorted(os.walk(self.dirpath)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    paths.append(path)

        sorted_paths = []
        ids = sorted(list(set([f.split('_')[0] for f in os.listdir(self.dirpath) if '.ipynb' not in f])))

        for id_ in sorted(ids):
            sorted_paths.append(self.find_paths_for_id(paths, id_, self.input_sequences))
        return sorted_paths

    def find_paths_for_id(self, imagepaths, id_, sequences):

        results = []

        for seq in sequences:

            imtag = id_ + '_' + seq + '.nii' # +'.' added to account for T1 vs T1gd
            images = [im for im in imagepaths if imtag in im]
            images = [im for im in images if 'ROI' not in im]

            if len(images) == 1:
                results.append(images[0])
            elif len(images) > 1:
                print('found %d images for %s - %s'%(len(images), seq, id_))
                for im in images:
                    print(im)
                print('keeping the first item :', images[0])
                results.append(images[0])
            else:
                print(seq, id_, '[]')

        return results

    def save_snapshot(self, images, x, y, z, label):
        slice_ = images[0][:, :, z]
        fig, ax = plt.subplots()
        ax.imshow(slice_)
        ax.scatter(x, y, s=200, c='red', marker='x')
        figname = os.path.join('./trainset_visuals/labeled', label + '.png')
        fig.savefig(figname)
        plt.close(fig)
        print(figname)

    def find_tagged_path(self, paths, tag):
        path = [f for f in paths if tag in f]
        if path:
            return path[0]

    def load_image(self, path):
        if path:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).T
        else:
            arr = np.full((256, 256, 256), 0.0)

        return arr

    def get_rows_with_labels(self):
        # load spreadsheet of coordinates
        labels_df = pd.read_csv(self.labels_csv)

        # get rows that have available labels
        indices = labels_df.index[labels_df[self.target].notna()]
        labels_df = labels_df.loc[indices, :]

        labels_df = self.clean_dataframe(labels_df)
        labels_df = self.change_to_classification(labels_df)
        return labels_df

    def clean_dataframe(self, labels_df):
        labels_df = labels_df.loc[:, ['AnonID', 'final_x', 'final_y', 'final_z', self.target]]

        return labels_df

    def change_to_classification(self, labels_df):
        # define thresholds for classification based on either median
        # or clinically meaningful thresholds  or manually set
        threshold_dct = {}

        if self.opt.criterion == 'median':

            threshold_dct[self.target] = np.median(labels_df[self.target].values)

        elif self.opt.criterion == 'clinical':

            threshold_dct['Ki67 LI'] = 0.1
            threshold_dct['SOX2 LI'] = 0.2
            threshold_dct['CD68LI'] = 0.4

        elif self.opt.criterion == 'input':

            threshold_dct[self.target] = self.opt.threshold

        else:
            print('Error!! high-low threshold not recognized. return')
            sys.exit()

        y = labels_df[self.target].values
        labels_df.loc[:, self.target + '_orig'] = y

        th = threshold_dct[self.target]
        y_class = [1 if v > th else 0 for v in y]

        labels_df.loc[:, self.target] = y_class
        print('\n%d positive,  %d negative samples'%(sum(y_class), len(y_class) - sum(y_class)))

        # 2 class classification needs a one hot encoder
        if self.num_classes > 1:

            y_1hot = np.array(y_class)
            y_1hot = (np.arange(self.num_classes) == y_1hot[:, None]).astype(np.float32)
            labels_df[self.target + '_1hot_1'] = -1
            labels_df[self.target + '_1hot_2'] = -1

            indices = list(labels_df.index.values)
            for i in range(len(indices)):
                labels_df.loc[indices[i], [self.target + '_1hot_1',
                                           self.target + '_1hot_2']] = y_1hot[i]

        return labels_df

    def get_images(self, id_):

        images = []
        paths = [p for p in self.paths if id_ in p[0]]
        rescale = tio.RescaleIntensity((0, 1), percentiles=(1, 99))

        if not paths:
            return None

        paths = paths[0]

        # find images for current id
        for seq in self.input_sequences:

            impath = [f for f in paths if seq + '.nii' in f]

            if impath:

                img = self.load_image(impath[0])
                if img.min() != img.max():
                    tensor = torch.from_numpy(img[None, :])
                    tensor = rescale(tensor)
                    img = tensor.numpy()[0, :, :, :]
                    img = img.astype(np.double)

            else:
                img = np.full((256, 256, 256), 0.5)

            images.append(img)

        return images

    def extract_roi(self, arr, coords):

        x, y, z = coords

        roi = arr[x - int(self.roisize/2): x + int(self.roisize/2),
                  y - int(self.roisize/2): y + int(self.roisize/2),
                  z]

        return roi


    def move_to_channels(self, rois, zeropad=False):
        # because we are using different rois as channels,
        # we can no longer have channels for each roi
        merged_roi = np.zeros((self.roisize, self.roisize, self.input_nc))

        for ind in range(len(rois)):
            roi = rois[ind]
            merged_roi[:, :, ind] = roi

        if zeropad:
            halfsize = int(self.roisize/2)
            padsize = []

            for ind in range(len(rois)):
                padsize.append((halfsize, halfsize))

            merged_roi = np.pad(merged_roi, tuple(padsize), 'constant')

        # it yells at you without transpose
        return merged_roi.T

    def process_item(self, id_, images, coordvals, label_1hotvals, augvals):

        '''extract roi at location.'''
        rois = []

        flipaxis, rot = augvals

        for img in images:

            roi = self.extract_roi(img, coordvals)

            if flipaxis > 0:
                roi = np.flip(roi, axis=flipaxis)

            if rot > 0:
                roi = rotate(roi, angle=rot, mode='wrap')

            rois.append(roi)

        img = self.move_to_channels(rois)
        self.images.append(img)

        self.labels.append(label_1hotvals)

    def make_dataset(self):
        '''Create the dataset in the format that it will be used,
        i.e. each item is an input to the model'''

        # prepare dataframe of labels
        labels_df = self.get_rows_with_labels()
        okids = list(set(labels_df['AnonID'].values))

        cols = list(labels_df.columns.values)
        onehotcols = [self.target + '_1hot_1', self.target + '_1hot_2']
        coordcols  = ['final_x', 'final_y', 'final_z']
        augcols    = ['flipaxis', 'rot']
        labelcols  = [self.target + '_orig', self.target]

        # add 4 aug cols to cols
        for v in augcols + onehotcols:
            if v not in cols:
                cols.append(v)

        print('total: %d samples of %d unique pts have %s labels\n' % (labels_df.shape[0],
                                                               len(okids),
                                                               self.target))

        # create a master dataframe for true, flipped, contralateral samples
        # for each biopsy loc we have :orig + 2flip + 5 rot + contralat : (9) version
        # if sox2 then also
        self.master_df = pd.DataFrame(columns=cols, index=np.arange(len(okids)*50))

        num_rots = 5
        ind2 = 0

        for id_ in okids:

            # get coords for this id
            rows = labels_df[labels_df['AnonID'] == id_]
            print(id_, ':', rows.shape[0], 'rows')

            images = self.get_images(id_)
            if not images:
                continue

            for index in rows.index.values:

                y, x, z = rows.loc[index, coordcols]
                labelvals = rows.loc[index, labelcols]
                label_1hotvals = rows.loc[index, onehotcols].values
                coordvals = [x, y, z] # flip x and y otherwise it ll be wrong

                # add flipped versions
                for flipaxis in [-1, 0, 1]: # no flip, flip x, flip y

                    rot = -1 # no rotation
                    augvals = [flipaxis, rot]

                    try:
                        self.process_item(id_, images, coordvals, label_1hotvals, augvals)

                        self.master_df.loc[ind2, onehotcols] = label_1hotvals
                        self.master_df.loc[ind2, 'AnonID'] = id_
                        self.master_df.loc[ind2, coordcols] = coordvals
                        self.master_df.loc[ind2, labelcols] = labelvals
                        self.master_df.loc[ind2, augcols] = augvals
                        ind2 += 1
                    except Exception as e:
                        print(e)

                # add rotations
                for num_rot in range(num_rots):

                    flipaxis = -1
                    rot = float(random.randint(-100, 100))/10.0
                    augvals = [flipaxis, rot]

                    try:
                        self.process_item(id_, images, coordvals, label_1hotvals, augvals)

                        self.master_df.loc[ind2, onehotcols] = label_1hotvals
                        self.master_df.loc[ind2, 'AnonID'] = id_
                        self.master_df.loc[ind2, coordcols] = coordvals
                        self.master_df.loc[ind2, labelcols] = labelvals
                        self.master_df.loc[ind2, augcols] = augvals
                        ind2 += 1

                    except Exception as e:
                        print(e)

                # add contralateral roi at 256-y for ki67 or sox2
                if self.target == 'Ki67 LI' or self.target == 'SOX2 LI':

                    y = 256 - y
                    coordvals = [x, y, z]

                    flipaxis = -1 # no flip
                    rot = -1 # no rotation
                    augvals = [flipaxis, rot]

                    labelvals = [-1, 0.0] # label orig unknown, label class = 0
                    label_1hotvals = [1.0, 0.0] # one hot version of label=0

                    try:
                        self.process_item(id_, images, coordvals, label_1hotvals, augvals)

                        self.master_df.loc[ind2, onehotcols] = label_1hotvals
                        self.master_df.loc[ind2, 'AnonID'] = id_
                        self.master_df.loc[ind2, coordcols] = coordvals
                        self.master_df.loc[ind2, labelcols] = labelvals
                        self.master_df.loc[ind2, augcols] = augvals
                        ind2 += 1

                    except Exception as e:
                        print(e)

        self.master_df.dropna(axis=0, how='all', inplace=True)


    def __getitem__(self, index):

        img = self.images[index]

        img_tensor = torch.from_numpy(img[None, :].astype(np.float32))
        rescale = tio.RescaleIntensity((0, 1), percentiles=(1, 99))
        img_tensor = rescale(img_tensor)

        label = np.array(self.labels[index]).astype(np.float32)
        label_tensor = torch.from_numpy(label)

        return {'tensor':img_tensor[0, :, :, :],
                'label' : label_tensor}
