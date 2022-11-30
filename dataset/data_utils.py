import io
import random
import tarfile
import logging
from urllib.parse import urlparse
from subprocess import PIPE, Popen
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import commons
from text import text_to_symbol


AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'curl -s -L {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix == 'txt':
                        example['txt'] = file_obj.read().decode('utf8').strip()
                    elif postfix == 'sid':
                        example['sid'] = file_obj.read().decode('utf8').strip()
                    elif postfix == 'spec':
                        data = io.BytesIO(file_obj.read())
                        spec = torch.load(data)
                        example['spec'] = spec
                    elif postfix in AUDIO_FORMAT_SETS:
                        audio, sampling_rate = torchaudio.load(file_obj)
                        # assert sampling_rate != xxx, f"The sampling rate must be {}"
                        example['wav'] = audio
                        example['sample_rate'] = sampling_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def data_filter(data, min_text_len=1, max_text_len=190, min_frame_len=32):
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'spec' in sample
        assert 'txt' in sample
        if len(sample['txt']) < min_text_len or len(sample['txt']) > max_text_len:
            continue
        if sample['spec'].size(1) < min_frame_len:
            continue
        yield sample


def tokenize(data, add_blank, symbol=None):
    for sample in data:
        text_norm = text_to_symbol(sample['txt'], symbol)
        if add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        sample['txt'] = text_norm
        yield sample


def data_shuffle(data, shuffle_size=10000):
    """ Local shuffle the data
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def data_sort(data, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['spec'].size(1))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['spec'].size(1))
    for x in buf:
        yield x


def batching(data, batch_size=16):
    """ Static batch the data by `batch_size`
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def padding(data):
    """ Padding the data into training data
    """
    for sample in data:
        assert isinstance(sample, list)
        spec_length = torch.tensor(
            [x['spec'].size(1) for x in sample], dtype=torch.int32)
        order = torch.argsort(spec_length, descending=True)

        sorted_text = [sample[i]['txt'] for i in order]
        text_lengths = torch.LongTensor([x.size(0) for x in sorted_text])
        sorted_spec = [sample[i]['spec'].transpose(0, 1) for i in order]
        spec_lengths = torch.LongTensor([x.size(0) for x in sorted_spec])
        sorted_wav = [sample[i]['wav'].transpose(0, 1) for i in order]
        wav_lengths = torch.LongTensor([x.size(0) for x in sorted_wav])

        padded_text = pad_sequence(sorted_text, batch_first=True, padding_value=0)
        padded_spec = pad_sequence(sorted_spec, batch_first=True, padding_value=0).transpose(1, 2)
        padded_wav = pad_sequence(sorted_wav, batch_first=True, padding_value=0).transpose(1, 2)

        yield (padded_text, text_lengths, padded_spec, spec_lengths, padded_wav, wav_lengths)
