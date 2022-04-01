import os.path
import random
import torch
from data.base_dataset import BaseDataset
import numpy as np
import SimpleITK as sitk
import torchio as tio
import matplotlib.pyplot as plt


class UnlabeledDataset(BaseDataset):

    def __len__(self):
        return len(self.images)

    def name(self):
        return 'UnlabeledDataset'

    def initialize(self, opt):

        print('\n', self.name())

        self.opt = opt
        self.roisize = opt.roisize
        self.input_nc = 2
        self.num_coords_percase = 20
        self.input_sequences = ['T1GD', 'T2']
        self.dirpath = os.path.join(opt.dataroot, 'unlabeled_skullstripped')

        self.images = []

        # get image paths
        self.paths = self.sort_paths()

        self.make_dataset()
        print('%d, # samples' % len(self.images))

    def find_tagged_path(self, paths, tag):
        path = [f for f in paths if tag in f[0]]
        if path:
            return path[0]

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
            sorted_paths.append(self.find_paths_for_id(paths, id_)[0])

        return sorted_paths

    def find_paths_for_id(self, imagepaths, id_):

        results = []
        for seq in self.input_sequences:

            imtag = id_ + '_' + seq
            images = [im for im in imagepaths if imtag in im]
            images = [im for im in images if 'ROI' not in im]

            d1img = [im for im in images if 'date1.' in im][0]
            d2img = [im for im in images if 'date2.' in im][0]

            if d1img and d2img:
                if results:
                    results[0][0].append(d1img)
                    results[0][1].append(d2img)

                else:
                    results.append([[d1img], [d2img]])

        return results

    def save_snapshot(self, images, x, y, z, label):

        slice_ = images[0][:, :, z]
        fig, ax = plt.subplots()

        ax.imshow(slice_)
        ax.scatter(x, y, s=200, c='white', marker='x')
        fig.savefig('./trainset_visuals/unlabeled/' + label + '.png')


    def get_insideT2_coords(self, id_):

        # get t2 roi for date1
        maskpath = os.path.join(self.dirpath,  id_ + '_T2_date1_ROI.nii.gz')

        if not os.path.exists(maskpath):

            # choose date 2 roi instead
            maskpath = maskpath.replace('date1', 'date2')

            if not os.path.exists(maskpath):
                # no T2 mask, return
                return None

        # read mask
        arr = sitk.GetArrayFromImage(sitk.ReadImage(maskpath)).T

        ys, xs, zs = np.where(arr > 0)
        indices = list(np.arange(len(xs)))

        # randomly select indices of elements in list xs
        rand_inds = list(random.sample(indices, self.num_coords_percase))
        rand_coords = [[xs[ind], ys[ind], zs[ind]] for ind in rand_inds]

        return rand_coords

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

    def get_rois(self, images, coords_):

        rois = []
        for img in images:
            roi = self.extract_roi(img, coords_)
            rois.append(roi)

        return rois

    def get_images(self, paths):

        images = []
        rescale = tio.RescaleIntensity((0, 1), percentiles=(1, 99))


        for impath in paths:

            imname = os.path.basename(impath)
            img = sitk.GetArrayFromImage(sitk.ReadImage(impath)).T

            if img.min() != img.max():

                tensor = torch.from_numpy(img[None, :])
                tensor = rescale(tensor)
                img = tensor.numpy()[0, :, :, :]

            images.append(img)

        return images

    def make_dataset(self):

        print('-found %d unlabeled repeat-scan pairs' % len(self.paths))

        okids = list(set([os.path.basename(pat[0][0]).split('_')[0] for pat in self.paths]))

        print('case ids:')
        for id_ in okids:

            id_imagepaths = [f for f in self.paths if id_ in f[0][0]][0]
            print(id_)
            date1_images = self.get_images(id_imagepaths[0])
            date2_images = self.get_images(id_imagepaths[1])
            if len(date1_images) != 2 and len(date2_images) != 2:
                print('did not find both time points for', id_)
                continue

            coords = self.get_insideT2_coords(id_)
            if not coords:
                continue

            for coord in coords:

                date1_rois = self.get_rois(date1_images, coord)
                date1_stackedimg  = self.move_to_channels(date1_rois)

                date2_rois = self.get_rois(date2_images, coord)
                date2_stackedimg  = self.move_to_channels(date2_rois)

                self.images.append([date1_stackedimg, date2_stackedimg])


    def __getitem__(self, index):

        date1img, date2img = self.images[index]

        date1_tensor = torch.from_numpy(date1img[None, :].astype(np.float32))
        date2_tensor = torch.from_numpy(date2img[None, :].astype(np.float32))

        rescale = tio.RescaleIntensity((0, 1), percentiles=(1, 99))

        date1_tensor = rescale(date1_tensor)
        date2_tensor = rescale(date2_tensor)

        return {
                'tensor1' : date1_tensor[0, :, :, :],
                'tensor2' : date2_tensor[0, :, :, :]
                }

