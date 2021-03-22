# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides core functions to load and split a dataset.
"""

# Imports
from collections import namedtuple, OrderedDict
import torch
import logging
import bisect
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit)
from sklearn.preprocessing import KBinsDiscretizer
# Global parameters
SetItem = namedtuple("SetItem", ["test", "train", "validation"])
DataItem = namedtuple("DataItem", ["inputs", "outputs", "labels"])


class ListTensors:
    def __init__(self, *tensor_list):
        self.list_tensors = list(tensor_list)

    def __getitem__(self, item):
        return self.list_tensors[item]

    def to(self, device, **kwargs):
        for i, e in enumerate(self.list_tensors):
            self.list_tensors[i] = e.to(device, **kwargs)
        return self.list_tensors

class DataManager(object):
    """ Data manager used to split a dataset in train, test and validation
    pytorch datasets.
    """

    def __init__(self, input_path, metadata_path, output_path=None, add_to_input=None,
                 labels=None, stratify_label=None, categorical_strat_label=True, custom_stratification=None,
                 N_train_max=None,  projection_labels=None, number_of_folds=10, batch_size=1, sampler=None,
                 in_features_transforms=None, input_transforms=None, output_transforms=None, labels_transforms=None,
                 stratify_label_transforms=None, data_augmentation=None, self_supervision=None, add_input=False,
                 patch_size=None, input_size=None, test_size=0.1, dataset=None, device='cpu', sep=',',
                 **dataloader_kwargs):
        """ Splits an input numpy array using memory-mapping into three sets:
        test, train and validation. This function can stratify the data.

        TODO: add how validation split is perform.
        TODO: fix case number_of_folds=1

        Parameters
        ----------
        input_path: str or list[str]
            the path to the numpy array containing the input tensor data
            that will be splited/loaded.
        metadata_path: str or list[str]
            the path to the metadata table in tsv format.
        output_path: str or list[str], default None
            the path to the numpy array containing the output tensor data
            that will be splited/loaded.
        add_to_input: list of str, default None
            list of features to add to the input
        labels: list of str, default None
            in case of classification/regression, the name of the column(s)
            in the metadata table to be predicted.
        stratify_label: str, default None
            the name of the column in the metadata table containing the label
            used during the stratification.
        categorical_strat_label: bool, default True
            is the stratification label a categorical or continuous variable ?
        custom_stratification: dict, default None
            same format as projection labels. It will split the dataset into train/test/val according
            to the stratification defined in the dict.
        N_train_max: int, default None
            set the max number of training samples that can be put in the training set. The stratification is made
            accordingly
        projection_labels: dict, default None
            selects only the data that match the conditions in the dict
            {<column_name>: <value>}.
        number_of_folds: int, default 10
            the number of folds that will be used in the cross validation.
        batch_size: int, default 1
            the size of each mini-batch.
        sampler: str in ["random", "weighted_random", "sequential"], default None
            Whether we use a weighted random sampler (to deal with imbalanced classes issue), random sampler (without
            replacement, to introduce shuffling in batches) or sequential (no shuffle)
        input_transforms, output_transforms: list of callable, default None
            transforms a list of samples with pre-defined transformations.
        data_augmentation: list of callable, default None
            transforms the training dataset input with pre-defined transformations on the fly during the training.
        self_supervision: a callable, default None
            applies a transformation to each input and generates a label
        add_input: bool, default False
            if true concatenate the input tensor to the output tensor.
        test_size: float, default 0.1
            should be between 0.0 and 1.0 and represent the proportion of the
            dataset to include in the test split.
        dataset: Dataset object, default None
            The Dataset used to create the DataLoader. It must be a subclass of <ArrayDataset>
        """
        assert input_path is None or type(input_path) == type(metadata_path)
        if output_path is not None:
            assert input_path is None or type(output_path) == type(input_path)

        input_path = [input_path] if type(input_path) == str else input_path
        metadata_path = [metadata_path] if type(metadata_path) == str else metadata_path
        output_path = [output_path] if output_path is not None else None

        assert input_path is None or len(input_path) == len(metadata_path)
        self.logger = logging.getLogger("pynet")

        if input_path is not None:
            for (i, m) in zip(input_path, metadata_path):
                self.logger.info('Correspondance {data} <==> {meta}'.format(data=i, meta=m))

            self.inputs = [np.load(p, mmap_mode='r') for p in input_path]
        else:
            self.inputs = None

        if output_path is not None:
            self.outputs = [np.load(p, mmap_mode='r') for p in output_path]

        all_df = [pd.read_csv(p, sep=sep) for p in metadata_path]
        assert self.inputs is None or np.all([len(i) == len(m) for (i,m) in zip(self.inputs, all_df)])

        df = pd.concat(all_df, ignore_index=True, sort=False)

        mask = DataManager.get_mask(
            df=df,
            projection_labels=projection_labels,
            check_nan=labels)

        mask_indices = DataManager.get_indices_from_mask(mask)

        # We should only work with masked data but we want to preserve the memory mapping so we are getting the right
        # index at the end (in __getitem__ of ArrayDataset)

        self.outputs, self.labels, self.stratify_label, self.features_to_add = (None, None, None, None)

        if labels is not None:
            if self_supervision is not None:
                raise ValueError("Impossible to set a label if self_supervision is on.")
            assert np.all(~df[labels][mask].isna())
            self.labels = df[labels].values.copy()
            self.labels = self.labels.squeeze()

        if stratify_label is not None:
            self.stratify_label = df[stratify_label].values.copy()
            # Apply the labels transform here as a mapping to the integer representation of the classes
            for i in mask_indices:
                label = self.stratify_label[i]
                for tf in (stratify_label_transforms or []):
                    label = tf(label)
                self.stratify_label[i] = label
            init_stratify_label_copy = self.stratify_label.copy()
            # If necessary, discretizes the labels
            if not categorical_strat_label:
                self.stratify_label[mask] = DataManager.discretize_continous_label(self.stratify_label[mask],
                                                                                   verbose=True)

        if add_to_input is not None:
            self.features_to_add = np.array([df[f].values for f in add_to_input]).transpose()

        self.metadata_path = metadata_path
        self.projection_labels = projection_labels
        self.number_of_folds = number_of_folds
        self.batch_size = batch_size
        self.input_transforms = input_transforms or []
        self.output_transforms = output_transforms or []
        self.labels_transforms = labels_transforms or []
        self.data_augmentation = data_augmentation or []
        self.self_supervision = self_supervision
        self.add_input = add_input
        self.data_loader_kwargs = dataloader_kwargs
        assert sampler in [None, "weighted_random", "random", "sequential"], "Unknown sampler: %s" % str(sampler)
        self.sampler = sampler

        dataset_cls = ArrayDataset if dataset is None else dataset
        assert issubclass(dataset_cls, ArrayDataset)


        if self.sampler == "weighted_random":
            if self.stratify_label is None:
                raise ValueError('Impossible to use the WeightedRandomSampler if no stratify label is available.')
            class_samples_count = [0 for _ in range(len(set(self.stratify_label[mask])))] # len == nb of classes
            for label in self.stratify_label[mask]:
                class_samples_count[label] += 1
            # Imbalanced weights in case of imbalanced classes issue
            self.sampler_weigths = 1. / torch.tensor(class_samples_count, dtype=torch.float)

        self.dataset = dict((key, [])
                            for key in ("train", "test", "validation"))

        if N_train_max is not None:
            assert custom_stratification is not None and \
                   {"train", "test"} <= set(custom_stratification.keys())

        # 1st step: split into train/test (get only indices)
        dummy_like_X_masked = np.ones(np.sum(mask))
        val_indices, train_indices, test_indices = (None, None, None)
        if custom_stratification is not None:
            if "validation" in custom_stratification and stratify_label is not None and N_train_max is None:
                print("Warning: impossible to stratify the data: validation+test set already defined ! ")
            train_mask, test_mask = (DataManager.get_mask(df, custom_stratification["train"]),
                                     DataManager.get_mask(df, custom_stratification["test"]))
            if "validation" in custom_stratification:
                val_mask = DataManager.get_mask(df, custom_stratification["validation"])
                val_mask &= mask
                val_indices = DataManager.get_indices_from_mask(val_mask)
                if N_train_max is None:
                    self.number_of_folds = 1

            train_mask &= mask
            test_mask &= mask
            train_indices = DataManager.get_indices_from_mask(train_mask)
            test_indices = DataManager.get_indices_from_mask(test_mask)

        elif stratify_label is not None:
            splitter = StratifiedShuffleSplit(
                n_splits=1, random_state=0, test_size=test_size)
            train_indices, test_indices = next(
                splitter.split(dummy_like_X_masked, self.stratify_label[mask]))
            train_indices = mask_indices[train_indices]
            test_indices = mask_indices[test_indices]
        else:
            if test_size == 1:
                train_indices, test_indices = (None, mask_indices)
            else:
                splitter = ShuffleSplit(
                    n_splits=1, random_state=0, test_size=test_size)
                train_indices, test_indices = next(splitter.split(dummy_like_X_masked))
                train_indices = mask_indices[train_indices]
                test_indices = mask_indices[test_indices]

        if train_indices is None:
            return

        assert len(set(train_indices) & set(test_indices)) == 0, 'Test set must be independent from train set'

        self.dataset["test"] = dataset_cls(
            self.inputs, test_indices, labels=self.labels,
            features_to_add=self.features_to_add,
            outputs=self.outputs, add_input=self.add_input,
            in_features_transforms=in_features_transforms,
            input_transforms = self.input_transforms,
            output_transforms = self.output_transforms,
            label_transforms = self.labels_transforms,
            self_supervision=self.self_supervision,
            patch_size=patch_size, input_size=input_size,
            concat_datasets=(self.inputs is not None),
            device=device)

        # 2nd step: split the training set into K folds (K-1 for training, 1
        # for validation, K times)

        if stratify_label is not None and not categorical_strat_label:
            # Recomputes the discretization for the training set to get a split train/val with finer statistics
            # (we do not assume that train+test has the same stats as train in case of custom stratification).
            self.stratify_label[train_indices] = \
                DataManager.discretize_continous_label(init_stratify_label_copy[train_indices], verbose=True)

        dummy_like_X_train = np.ones(len(train_indices))

        if N_train_max is not None:
            Splitter = ShuffleSplit if stratify_label is None else StratifiedShuffleSplit
            kfold_splitter = Splitter(n_splits=self.number_of_folds,
                                      train_size=float(N_train_max/len(train_indices)), random_state=0)
            strat_indices = np.array(self.stratify_label[train_indices], dtype=np.int32) \
                if stratify_label is not None else None
            gen = kfold_splitter.split(dummy_like_X_train, strat_indices)
            if val_indices is not None:
                gen = [(train_indices[tr], val_indices) for (tr, _) in gen]
            else:
                gen = [(train_indices[tr], train_indices[val]) for (tr, val) in gen]
        elif val_indices is not None:
            gen = [(train_indices, val_indices)]
        else:
            if self.number_of_folds > 1:
                Splitter = KFold if stratify_label is None else StratifiedKFold
                kfold_splitter = Splitter(n_splits=self.number_of_folds)
                strat_indices = np.array(self.stratify_label[train_indices], dtype=np.int32) \
                    if stratify_label is not None else None
                gen = kfold_splitter.split(dummy_like_X_train, strat_indices)
                gen = [(train_indices[tr], train_indices[val]) for (tr, val) in gen]
            else:
                gen = [(train_indices, [])]


        for fold_train_index, fold_val_index in gen:
            assert len(set(fold_val_index) & set(fold_train_index)) == 0, \
                'Validation set must be independant from test set'

            train_dataset = dataset_cls(
                self.inputs, fold_train_index,
                labels=self.labels, outputs=self.outputs,
                features_to_add=self.features_to_add,
                add_input=self.add_input,
                in_features_transforms=in_features_transforms,
                input_transforms=self.data_augmentation+self.input_transforms,
                output_transforms=self.data_augmentation+self.output_transforms,
                label_transforms=self.labels_transforms,
                self_supervision=self.self_supervision,
                patch_size=patch_size, input_size=input_size,
                concat_datasets=(self.inputs is not None),
                device=device)
            val_dataset = dataset_cls(
                self.inputs, fold_val_index,
                labels=self.labels, outputs=self.outputs,
                features_to_add=self.features_to_add,
                add_input=self.add_input,
                in_features_transforms=in_features_transforms,
                input_transforms=self.input_transforms,
                output_transforms=self.output_transforms,
                label_transforms=self.labels_transforms,
                self_supervision=self.self_supervision,
                patch_size=patch_size, input_size=input_size,
                concat_datasets=(self.inputs is not None),
                device=device
            )
            self.dataset["train"].append(train_dataset)
            self.dataset["validation"].append(val_dataset)

    @staticmethod
    def discretize_continous_label(labels, verbose=False):
        # Get an estimation of the best bin edges. 'Sturges' is conservative for pretty large datasets (N>1000).
        bin_edges = np.histogram_bin_edges(labels, bins='sturges')
        if verbose:
            print('Global histogram:\n', np.histogram(labels, bins=bin_edges, density=False), flush=True)
        # Discretizes the values according to these bins
        discretization = np.digitize(labels, bin_edges[1:], right=True)
        if verbose:
            print('Bin Counts after discretization:\n', np.bincount(discretization), flush=True)
        return discretization

    @staticmethod
    def get_indices_from_mask(mask):
        return np.arange(len(mask))[mask]

    def __getitem__(self, item):
        """ Return the requested item.

        Returns
        -------
        item: Dataset or list of Dataset
            the requested set of data: test, train or validation.
        """
        if item not in ("train", "test", "validation"):
            raise ValueError("Unknown set! Must be 'train', 'test' or "
                             "'validation'.")
        return self.dataset[item]

    def collate_fn(self, list_samples):
        """ After fetching a list of samples using the indices from sampler,
        the function passed as the collate_fn argument is used to collate lists
        of samples into batches.

        A custom collate_fn is used here to apply the transformations.

        See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn.
        """
        data = OrderedDict()
        for key in ("inputs", "outputs", "labels"):
            if len(list_samples) == 0 or getattr(list_samples[-1], key) is None:
                data[key] = None
            else:
                if key == "inputs" and self.features_to_add is not None:
                    input_ = torch.stack([torch.as_tensor(getattr(s, key)[0], dtype=torch.float) for s in list_samples], dim=0)
                    features = torch.stack([torch.as_tensor(getattr(s, key)[1], dtype=torch.float) for s in list_samples], dim=0)
                    data[key] = ListTensors(input_, features)
                else:
                    data[key] = torch.stack([torch.as_tensor(getattr(s, key), dtype=torch.float) for s in list_samples], dim=0)
        if data["labels"] is not None:
            data["labels"] = data["labels"].type(torch.FloatTensor)
        return DataItem(**data)

    def get_dataloader(self, train=False, validation=False, test=False,
                       fold_index=0):
        """ Generate a pytorch DataLoader.

        Parameters
        ----------
        train: bool, default False
            return the dataloader over the train set.
        validation: bool, default False
            return the dataloader over the validation set.
        test: bool, default False
            return the dataloader over the test set.
        fold_index: int, default 0
            the index of the fold to use for the training

        Returns
        -------
        loaders: list of DataLoader
            the requested data loaders.
        """
        _test, _train, _validation, sampler = (None, None, None, None)
        if test:
            _test = DataLoader(
                self.dataset["test"], batch_size=self.batch_size,
                collate_fn=self.collate_fn, **self.data_loader_kwargs)
        if train:
            if self.sampler == "weighted_random":
                indices = self.dataset["train"][fold_index].indices
                samples_weigths = self.sampler_weigths[np.array(self.stratify_label[indices], dtype=np.int32)]
                sampler = WeightedRandomSampler(samples_weigths, len(indices), replacement=True)
            elif self.sampler == "random":
                sampler = RandomSampler(self.dataset["train"][fold_index])
            elif self.sampler == "sequential":
                sampler = SequentialSampler(self.dataset["train"][fold_index])
            _train = DataLoader(
                self.dataset["train"][fold_index], batch_size=self.batch_size, sampler=sampler,
                collate_fn=self.collate_fn, **self.data_loader_kwargs)
        if validation:
            _validation = DataLoader(
                self.dataset["validation"][fold_index],
                batch_size=self.batch_size, collate_fn=self.collate_fn,
                **self.data_loader_kwargs)
        return SetItem(test=_test, train=_train, validation=_validation)

    @staticmethod
    def get_mask(df, projection_labels=None, check_nan=None):
        """ Filter a table.

        Parameters
        ----------
        df: a pandas DataFrame
            a table data.
        projection_labels: dict, default None
            selects only the data that match the conditions in the dict
            {<column_name>: <value>}.
        check_nan: list of str, default None
            check if there is nan in the selected columns. Select only the rows without nan
        Returns
        -------
        mask: a list of boolean values
        """

        mask = np.ones(len(df), dtype=np.bool)
        if projection_labels is not None:
            for (col, val) in projection_labels.items():
                if isinstance(val, list):
                    mask &= getattr(df, col).isin(val)
                elif val is not None:
                    mask &= getattr(df, col).eq(val)
        if check_nan is not None:
            for col in check_nan:
                mask &= ~getattr(df, col).isna()
        return mask


    def dump_augmented_data(self, N_per_class, output_path, output_path_df):
        ## It takes all the dataset and computes, for each class sample, transformations that preserve the class
        #  distribution as homogeneously as possible
        def apply_transforms(obj, tfs):
            obj_tf = obj
            for tf in tfs:
                obj_tf = tf(obj)
            return obj_tf

        # First, get the mask to consider only relevant data for our application
        df = pd.read_csv(self.metadata_path, sep="\t")
        mask = DataManager.get_mask(df=df, projection_labels=self.projection_labels)

        labels_mapping = {l: apply_transforms(l, self.labels_transforms) for l in set(self.labels[mask])}

        class_repartition = [0 for _ in range(len(set(labels_mapping.values())))] # len == nb of classes
        for i, label in enumerate(self.labels[mask]):
            label = labels_mapping[label]
            class_repartition[label] += 1

        n_classes = len(class_repartition)
        if isinstance(N_per_class, int):
            N_per_class = [N_per_class for _ in class_repartition]
        elif isinstance(N_per_class, list):
            assert len(N_per_class) == n_classes

        missing_samples_per_class = [N_per_class[i] for i in range(n_classes)]
        adding_samples_per_class = [(missing_samples_per_class[i]-1)//class_repartition[i] + 1
                                    if missing_samples_per_class[i] > 0 else 0 for i in range(n_classes)]

        len_X_augmented = np.sum(missing_samples_per_class)

        X_to_dump = np.memmap(output_path, dtype='float32', mode='w+', shape=(len_X_augmented,)+self.inputs[0].shape)
        df_to_dump = np.zeros(shape=(len_X_augmented, len(df.columns)), dtype=object)
        # For each class, add the missing samples with the data_augmentation_transforms
        count = 0
        pbar = tqdm(total=np.sum(mask), desc="Input images processed")
        for i in DataManager.get_indices_from_mask(mask):
            pbar.update()
            sample = self.inputs[i]
            label = labels_mapping[self.labels[i]]
            if missing_samples_per_class[label] > 0:
                for j in range(adding_samples_per_class[label]):
                    if missing_samples_per_class[label] > 0:
                        x_transformed = sample
                        for tf in self.data_augmentation:
                            x_transformed = tf(x_transformed)
                        X_to_dump[count] = x_transformed
                        df_to_dump[count] = df.values[i]
                        count += 1
                        missing_samples_per_class[label] -= 1

        df_to_dump = pd.DataFrame(df_to_dump, columns=df.columns)
        df_to_dump.to_csv(output_path_df, index=False, sep='\t')

class ArrayDataset(Dataset):
    """ A dataset based on numpy array.
    """
    def __init__(self, inputs, indices, labels=None, outputs=None, features_to_add=None,
                 add_input=False, in_features_transforms=None,
                 input_transforms=None, output_transforms=None,
                 label_transforms=None, self_supervision=None,
                 patch_size=None, input_size=None, concat_datasets=False, device='cpu'):
        """ Initialize the class.

        Parameters
        ----------
        inputs: numpy array or list of numpy array
            the input data.
        indices: iterable of int
            the list of indices that is considered in this dataset.
        labels: DataFrame or numpy array
        features_to_add: list of pd.Series
            list of features to add to each input
        outputs: numpy array or list of numpy array
            the output data.
        add_input: bool, default False
            if set concatenate the input data to the output (useful with
            auto-encoder).
        self_supervision: callable, default None
            if set, the transformation to apply to each input that will generate a label
        patch_size: tuple, default None
            if set, return only patches of the input image
        concat_datasets: bool, default False
            whether to consider a list of inputs/outputs as a list of multiple datasets or a unique dataset
        """

        self.inputs = inputs
        self.labels = labels
        self.outputs = outputs
        self.device = device
        self.indices = indices
        self.concat_datasets = concat_datasets
        self.add_input = add_input
        self.patch_size = patch_size
        self.input_size = input_size
        self.features_to_add = features_to_add
        if self.patch_size is not None:
            assert np.array(self.patch_size).shape == np.array(self.input_size).shape
            self.nb_patches_by_img = np.product(np.array(self.input_size) // np.array(self.patch_size))
            self.input_cached, self.output_cached, self.label_cached, self.input_indx_cached = None, None, None, None
        self.in_features_transforms = in_features_transforms or []
        self.input_transforms = input_transforms or []
        self.output_transforms = output_transforms or []
        self.labels_transforms = label_transforms or []
        self.self_supervision = self_supervision
        self.output_same_as_input = (self.add_input and self.outputs is None)
        if self.add_input and self.outputs is None:
            self.outputs = self.inputs
            self.add_input = False
        if self.concat_datasets:
            self.cumulative_sizes = np.cumsum([len(inp) for inp in self.inputs])

        if self.labels is not None and self.inputs is not None:
            assert (self.concat_datasets and self.cumulative_sizes[-1] == len(self.labels)) or \
                   len(self.inputs) == len(self.labels)
        if self.outputs is not None and self.inputs is not None:
            assert len(self.inputs) == len(self.outputs)

    def __getitem__(self, item):
        """ Return the requested item.

        Returns
        -------
        item: namedtuple
            a named tuple containing 'inputs', 'outputs', and 'labels' data.
        """
        if isinstance(item, int):
            concat_axis = 0
        else:
            concat_axis = 1

        if self.patch_size is not None:
            offset = item % self.nb_patches_by_img
            item = item // self.nb_patches_by_img
            if self.input_indx_cached == item:
                # Retrieve directly the input (and eventually the output)
                indx = np.unravel_index(offset, np.array(self.input_size) // np.array(self.patch_size))
                _inputs = self.input_cached[indx]
                _outputs, _labels = self.output_cached, self.label_cached
                if self.output_same_as_input:
                    _outputs = self.output_cached[indx]
                return DataItem(inputs=_inputs, outputs=_outputs, labels=_labels)

        idx = self.indices[item]
        _outputs = None
        if self.concat_datasets:
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            sample_idx = idx - self.cumulative_sizes[dataset_idx-1] if dataset_idx > 0 else idx
            _inputs = self.inputs[dataset_idx][sample_idx]
            if self.outputs is not None:
                _outputs = self.outputs[dataset_idx][sample_idx]
        else:
            _inputs = self.inputs[idx]
            if self.outputs is not None:
                _outputs = self.outputs[idx]

        if self.outputs is not None and self.add_input:
            _outputs = np.concatenate((_outputs, _inputs), axis=concat_axis)

        _labels = None
        if self.labels is not None: # Particular case in which we can deal with strings before transformations...
            _labels = self.labels[idx]

        # Apply the transformations to the data
        for tf in self.input_transforms:
            _inputs = tf(_inputs)
        if _outputs is not None:
            for tf in self.output_transforms:
                _outputs = tf(_outputs)
        if _labels is not None:
            for tf in self.labels_transforms:
                _labels = tf(_labels)

        if self.self_supervision is not None:
            _inputs, _labels = self.self_supervision(_inputs)

        # Eventually, get only one patch of the input (and one patch of the corresponding output if add_input==True)
        if self.patch_size is not None:
            self.input_indx_cached = item
            from skimage.util.shape import view_as_blocks
            # from a flat index, convert it to an nd-array index
            indx = np.unravel_index(offset, np.array(self.input_size) // np.array(self.patch_size))

            # Store everything in a cache to avoid useless computations...
            self.input_cached = view_as_blocks(_inputs, tuple(self.patch_size))
            self.output_cached = _outputs
            if self.output_same_as_input:
                self.output_cached = _outputs, tuple(self.patch_size)
            self.label_cached = _labels

            _inputs = self.input_cached[indx]
            if self.output_same_as_input:
                _outputs = self.output_cached[indx]

        # Finally, add the other input features, if any
        if self.features_to_add is not None:
            _features = self.features_to_add[self.indices[item]]
            if self.in_features_transforms is not None:
                for tf in self.in_features_transforms:
                    _features = tf(_features)
            _inputs = [_inputs, _features]

        return DataItem(inputs=_inputs, outputs=_outputs, labels=_labels)

    def __len__(self):
        """ Return the length of the dataset.
        """
        if self.patch_size is not None:
            return len(self.indices) * self.nb_patches_by_img
        return len(self.indices)
