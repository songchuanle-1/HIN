# coding: utf8
from collections import Counter, OrderedDict
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.folder import default_loader
from itertools import chain
import six
import torch
import numpy as np
import h5py
import os
import warnings
import shutil

from .vocab import Vocab
from .utils import get_tokenizer


class RawField(object):
    """ Defines a general datatype.

    Every dataset consists of one or more types of data. For instance,
    a machine translation dataset contains paired examples of text, while
    an image captioning dataset contains images and texts.
    Each of these types of data is represented by a RawField object.
    An RawField object does not assume any property of the data type and
    it holds parameters relating to how a datatype should be processed.

    Attributes:
        preprocessing: The Pipeline that will be applied to examples
            using this field before creating an example.
            Default: None.
        postprocessing: A Pipeline that will be applied to a list of examples
            using this field before assigning to a batch.
            Function signature: (batch(list)) -> object
            Default: None.
    """

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        """ Preprocess an example if the `preprocessing` Pipeline is provided. """
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        """ Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
                postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)


class OtherField(RawField):
    def __init__(self):
        self.kkk = ['image_id','caption']

    def preprocess(self, x):
        return x
        # x = x['caption']
        # if six.PY2 and isinstance(x, six.string_types) and not isinstance(x, six.text_type):
        #     x = six.text_type(x, encoding='utf-8')
        # if self.lower:
        #     x = six.text_type.lower(x)
        # x = self.tokenize(x.rstrip('\n'))
        # if self.remove_punctuation:
        #     x = [w for w in x if w not in self.punctuations]
        # if self.preprocessing is not None:
        #     x =  self.preprocessing(x)

        # if self.reverse:
        #     return x, list(reversed(x))
        # else:
        #     return x


    def process(self, batch):
        # images_ids = []
        # captions = []
        # print(batch)
        return batch
        # for i in range(len(batch)):
        #     images_ids.append(batch[i]['image_id'])
        #     captions.append(batch[i]['caption'])
        # return [images_ids,captions]


class Merge(RawField):
    def __init__(self, *fields):
        super(Merge, self).__init__()
        self.fields = fields

    def preprocess(self, x):
        return tuple(f.preprocess(x) for f in self.fields)

    def process(self, batch, *args, **kwargs):
        if len(self.fields) == 1:
            batch = [batch, ]
        else:
            batch = list(zip(*batch))

        out = list(f.process(b, *args, **kwargs) for f, b in zip(self.fields, batch))
        return out


class ImageDetectionsField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, feature_type='butd', detections_path=None, max_detections=100,
                 with_pe=False, sort_by_prob=False, load_in_tmp=False, global_feature=False):
        self.max_detections = max_detections
        self.detections_path = detections_path
        self.feature_type = feature_type
        self.sort_by_prob = sort_by_prob
        self.with_pe = with_pe
        self.global_feature = global_feature

        tmp_detections_path = os.path.join('/tmp', os.path.basename(detections_path))

        if load_in_tmp:
            if not os.path.isfile(tmp_detections_path):
                if shutil.disk_usage("/tmp")[-1] < os.path.getsize(detections_path):
                    warnings.warn('Loading from %s, because /tmp has no enough space.' % detections_path)
                else:
                    warnings.warn("Copying detection file to /tmp")
                    shutil.copyfile(detections_path, tmp_detections_path)
                    warnings.warn("Done.")
                    self.detections_path = tmp_detections_path
            else:
                self.detections_path = tmp_detections_path

        available_features = ['butd', 'clip', 'vinvl', 'tokens']
        assert self.feature_type in available_features, \
               "region feature not supported, please select ['butd', 'clip', 'vinvl', 'tokens']"

        if self.feature_type in ['butd', 'vinvl', 'clip', 'tokens']:
            self.f = h5py.File(self.detections_path, 'r')

        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        image_id, split, orig_size = x['image_id'], x['split'], x['orig_size']
        try:
            if self.feature_type in ['butd', 'vinvl']:
                precomp_data = torch.from_numpy(self.f['%d_features' % image_id][()])
                # boxes = torch.from_numpy(self.f['%d_boxes' % image_id][()])
                # print(boxes)
                if self.with_pe:
                    boxes = torch.from_numpy(self.f['%d_boxes' % image_id][()])
                    if len(boxes):
                        precomp_data = precomp_data[:len(boxes),:]

                if self.sort_by_prob:
                    idxs = torch.from_numpy(np.argsort(np.max(self.f['%d_cls_prob' % image_id][()], -1))[::-1])
                    precomp_data = precomp_data[idxs]
                    if self.with_pe:
                        boxes = boxes[idxs]

            elif self.feature_type == 'clip':
                precomp_data = torch.from_numpy(self.f['%d_features' % image_id][()])
                if self.global_feature:
                    global_feature = torch.from_numpy(self.f['%d_global' % image_id][()])
                    return precomp_data, global_feature
                return precomp_data
            
            elif self.feature_type == 'tokens':
                precomp_data = torch.from_numpy(self.f['%d_tokens' % image_id][()])
                return precomp_data

            if self.with_pe:
                size = torch.tensor(orig_size).repeat(len(boxes), 2)
                relative_boxes = boxes / size
                
        except KeyError:
            warnings.warn('Could not find detections for %d' % image_id)
            precomp_data = torch.rand(10,2048)
            relative_boxes = torch.rand((10, 4))

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = torch.cat([precomp_data, torch.zeros((delta, precomp_data.shape[1]))], 0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_detections]

        # if self.with_pe:
        #     delta_boxes = self.max_detections - len(relative_boxes)
        #     if delta_boxes > 0:
        #         relative_boxes = torch.cat([relative_boxes, torch.zeros((delta_boxes, relative_boxes.shape[1]))], 0)
        #     elif delta_boxes < 0:
        #         relative_boxes = relative_boxes[:self.max_detections]
        if self.with_pe:
            delta_boxes = self.max_detections - len(relative_boxes)
            if delta_boxes > 0:
                relative_boxes = torch.cat([relative_boxes, torch.zeros((delta_boxes, relative_boxes.shape[1]))], 0)
            elif delta_boxes < 0:
                relative_boxes = relative_boxes[:self.max_detections]
            return (precomp_data, relative_boxes)

        return precomp_data

class DualImageField(RawField):
    def __init__(self, clip_path, vinvl_path, preprocessing=None, postprocessing=None, max_detections=100, global_feature=False,
                 with_pe=False, sort_by_prob=False, load_in_tmp=False):

        self.clip_field = ImageDetectionsField(preprocessing, postprocessing, 'clip', clip_path, global_feature=global_feature)
        self.vinvl_field = ImageDetectionsField(preprocessing, postprocessing, 'vinvl', vinvl_path, 
                                                max_detections, with_pe, sort_by_prob, load_in_tmp)
        self.global_feature = global_feature
        super().__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        region_features =  self.vinvl_field.preprocess(x)
        if self.global_feature:
            grid_features, global_feature = self.clip_field.preprocess(x)
            return (grid_features, region_features, global_feature)
        else:
            grid_features = self.clip_field.preprocess(x)
            return (grid_features, region_features)

class TextField(RawField):
    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    def __init__(self, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False, tokenize=(lambda s: s.split()),
                 remove_punctuation=False, include_lengths=False, batch_first=True, pad_token="<pad>",
                 unk_token="<unk>", pad_first=False, truncate_first=False, vectors=None, nopoints=True, reverse=False):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.remove_punctuation = remove_punctuation
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.vocab = None
        self.vectors = vectors
        self.reverse = reverse
        if nopoints:
            self.punctuations.append("..")

        super(TextField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        # x = x['caption']
        if six.PY2 and isinstance(x, six.string_types) and not isinstance(x, six.text_type):
            x = six.text_type(x, encoding='utf-8')
        if self.lower:
            x = six.text_type.lower(x)
        x = self.tokenize(x.rstrip('\n'))
        if self.remove_punctuation:
            x = [w for w in x if w not in self.punctuations]
        if self.preprocessing is not None:
            x =  self.preprocessing(x)

        if self.reverse:
            return x, list(reversed(x))
        else:
            return x

    def process(self, batch, device=None):
        if self.reverse:
            batch = list(zip(*batch))
            padded_1 = self.pad(batch[0])
            padded_2 = self.pad(batch[1], reverse=True)
            tensor_1 = self.numericalize(padded_1, device=device)
            tensor_2 = self.numericalize(padded_2, device=device)
            return tensor_1, tensor_2
            # padded = self.pad(batch, reverse=True)
            # tensor = self.numericalize(padded, device=device)
            # return tensor
        else:
            padded = self.pad(batch)
            tensor = self.numericalize(padded, device=device)
            return tensor

    def build_vocab(self, *args, **kwargs):
        from .dataset import Dataset
        
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)

        for data in sources:
            for x in data:
                x = self.preprocess(x)
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))

        specials = list(OrderedDict.fromkeys([
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None]))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def pad(self, minibatch, reverse=False):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just
        returns the padded list.
        """
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            elif reverse:
                padded.append(
                    ([] if self.eos_token is None else [self.eos_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.init_token is None else [self.init_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a list of Variables.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)

            var = torch.tensor(arr, dtype=self.dtype, device=device)
        else:
            if self.vectors:
                arr = [[self.vectors[x] for x in ex] for ex in arr]
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            arr = [numericalization_func(x) if isinstance(x, six.string_types)
                   else x for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

            var = torch.cat([torch.cat([a.unsqueeze(0) for a in ar]).unsqueeze(0) for ar in arr])

        # var = torch.tensor(arr, dtype=self.dtype, device=device)
        if not self.batch_first:
            var.t_()
        var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ], join_words)[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ], join_words)[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[int(wi)]
                if word == self.eos_token:
                    break
                caption.append(word)
            if join_words:
                caption = ' '.join(caption)
            captions.append(caption)
        return captions