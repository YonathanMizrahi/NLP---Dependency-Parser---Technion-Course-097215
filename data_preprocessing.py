import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict

torch.manual_seed(0)

UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
ROOT_TOKEN = "<root>"
SPECIAL_TOKENS = [ROOT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]

class DataReader:
    """ Reads the data from the requested file and hold it's components. """

    def __init__(self, word_dict, pos_dict, file_path, competition=False):
        self.competition = competition
        self.file_path = file_path
        self.words_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        cur_sentence_word = [ROOT_TOKEN]
        cur_sentence_pos = [ROOT_TOKEN]
        cur_sentence_headers = [-1]

        with open(self.file_path, 'r') as f:
            for line in f:
                split_line = line.split('\t')
                if len(split_line) == 1:  # the end of a sentence denotes by \n line.
                    self.sentences.append((cur_sentence_word, cur_sentence_pos, cur_sentence_headers))
                    cur_sentence_word = [ROOT_TOKEN]
                    cur_sentence_pos = [ROOT_TOKEN]
                    cur_sentence_headers = [-1]
                    continue
                if not self.competition:
                    word, pos_tag, head = split_line[1], split_line[3], int(split_line[6])
                else:
                    word, pos_tag, head = split_line[1], split_line[3], -2
                cur_sentence_word.append(word)
                cur_sentence_pos.append(pos_tag)
                cur_sentence_headers.append(head)

    def get_num_sentences(self):
        """returns num of sentences in data."""
        return len(self.sentences)


class PosDataset(Dataset):
    def __init__(self, word_dict, pos_dict, file_path, padding=False, word_embeddings=None, competition=False):
        super().__init__()
        self.file_path = file_path
        self.data_reader = DataReader(word_dict, pos_dict, self.file_path, competition)
        self.vocab_size = len(self.data_reader.words_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if word_embeddings:
            self.words_idx_mappings, self.idx_words_mappings, self.words_vectors = word_embeddings
        else:
            # word_idx_mapping:
            all_words = list(self.data_reader.words_dict.keys())
            self.words_idx_mappings = dict(map(reversed, enumerate(all_words)))
            self.idx_words_mappings = all_words
            self.words_vectors = None

            # idx_pox_mapping
            all_pos = list(self.data_reader.pos_dict.keys())
            self.pos_idx_mappings = dict(map(reversed, enumerate(all_pos)))
            self.idx_pos_mappings = all_pos
            self.pos_vectors = None

        self.pad_idx = self.words_idx_mappings.get(PAD_TOKEN)
        self.unknown_idx = self.words_idx_mappings.get(UNKNOWN_TOKEN)
        self.sentence_lens = [len(sentence[0]) for sentence in self.data_reader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, head_embed_idx, sentence_len = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, head_embed_idx, sentence_len

    def get_words_embeddings(self):
        return self.words_idx_mappings, self.idx_words_mappings, self.words_vectors

    def get_pos_embeddings(self):
        return self.pos_idx_mappings, self.idx_pos_mappings, self.pos_vectors

    def convert_sentences_to_dataset(self, padding):
        sentence_words_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_headers_idx_list = list()
        sentence_len_list = self.sentence_lens

        for sentence_idx, sentence in enumerate(self.data_reader.sentences):
            words_idx_list = []
            pos_idx_list = []
            headers_idx_list = []

            for word, pos_tag, header in zip(sentence[0], sentence[1], sentence[2]):

                headers_idx_list.append(header)
                if word in self.data_reader.words_dict:
                    words_idx_list.append(self.words_idx_mappings.get(word))
                else:
                    words_idx_list.append(self.unknown_idx)
                if pos_tag in self.data_reader.pos_dict:
                    pos_idx_list.append(self.pos_idx_mappings.get(pos_tag))
                else:
                    pos_idx_list.append(self.unknown_idx)

            if padding:
                while len(words_idx_list) < self.max_seq_len:
                    words_idx_list.append(self.pad_idx)
                    pos_idx_list.append(self.pad_idx)
                    headers_idx_list.append(self.pad_idx)
                sentence_words_idx_list.append(words_idx_list)
                sentence_pos_idx_list.append(pos_idx_list)
                sentence_headers_idx_list.append(headers_idx_list)
            else:
                sentence_words_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False).to(self.device))
                sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False).to(self.device))
                sentence_headers_idx_list.append(torch.tensor(headers_idx_list, dtype=torch.long, requires_grad=False).to(self.device))

        if padding:
            all_sentence_words_idx = torch.tensor(sentence_words_idx_list, dtype=torch.long, requires_grad=False).to(self.device, non_blocking=True)
            all_sentence_tags_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long, requires_grad=False).to(self.device, non_blocking=True)
            all_sentence_labels_idx = torch.tensor(sentence_headers_idx_list, dtype=torch.long, requires_grad=False).to(self.device, non_blocking=True)
            all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False).to(self.device, non_blocking=True)
            return TensorDataset(all_sentence_words_idx, all_sentence_tags_idx, all_sentence_labels_idx, all_sentence_len)
        else:
            return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_words_idx_list,
                                                                         sentence_pos_idx_list,
                                                                         sentence_headers_idx_list,
                                                                         sentence_len_list))}


def get_vocabs(list_of_paths):
    """
    This function creates two ordered dictionaries, one for words and one for pos, for a given list of paths.
    It parses the paths for words and parts of speech, and appends them to the dictionaries, with the number of occurrences as values.
    The dictionaries words_dict pos_dict and are then returned.
    """
    words_dict = OrderedDict([(PAD_TOKEN, 1), (ROOT_TOKEN, 1), (UNKNOWN_TOKEN, 1)])
    pos_dict = OrderedDict([(PAD_TOKEN, 1), (ROOT_TOKEN, 1), (UNKNOWN_TOKEN, 1)])

    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                split_line = line.split('\t')
                if len(split_line) == 1:
                    continue
                word, pos_tag = split_line[1], split_line[3]
                if word in words_dict:
                    words_dict[word] = words_dict[word] + 1
                else:
                    words_dict[word] = 1
                if pos_tag in pos_dict:
                    pos_dict[pos_tag] = pos_dict[pos_tag] + 1
                else:
                    pos_dict[pos_tag] = 1

    return words_dict, pos_dict





