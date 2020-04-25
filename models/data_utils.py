import json
import csv
import os
import copy 

class InputExample:
    """Simple object to encapsulate each data example"""
    def __init__(self, src, trg, 
                 src_g, trg_g):    
        self.src = src
        self.trg = trg
        self.src_g = src_g
        self.trg_g = trg_g
    
    def __repr__(self):
        return str(self.to_json_str())
    
    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

class RawDataset:
    """Encapsulates the raw examples in InputExample objects"""
    def __init__(self, data_dir):
        self.train_examples = self.get_train_examples(data_dir)
        self.dev_examples = self.get_dev_examples(data_dir)
        self.test_examples = self.get_dev_examples(data_dir)
        
    def create_examples(self, src_path, trg_path):
        
        src_txt = self.get_txt_examples(src_path)
        src_gender_labels = self.get_labels(src_path + '.label')
        trg_txt = self.get_txt_examples(trg_path)
        trg_gender_labels = self.get_labels(trg_path + '.label')
        
        examples = []
        
        for i in range(len(src_txt)):
            src = src_txt[i].strip()
            trg = trg_txt[i].strip()
            src_g = src_gender_labels[i].strip()
            trg_g = trg_gender_labels[i].strip()
            input_example = InputExample(src, trg, src_g, trg_g)
            examples.append(input_example)
        
        return examples
    
    def get_labels(self, data_dir):
        with open(data_dir) as f:
            return f.readlines()
        
    def get_txt_examples(self, data_dir):
        with open(data_dir, encoding='utf8') as f:
            return f.readlines()
    
    def get_train_examples(self, data_dir):
        """Reads the train examples of the dataset"""
        return self.create_examples(os.path.join(data_dir, 'D-set-train.arin'), 
                                    os.path.join(data_dir, 'D-set-train.ar.M'))
    
    def get_dev_examples(self, data_dir):
        """Reads the dev examples of the dataset"""
        return self.create_examples(os.path.join(data_dir, 'D-set-dev.arin'), 
                                    os.path.join(data_dir, 'D-set-dev.ar.M'))
    
    def get_test_examples(self, data_dir):
        """Reads the test examples of the dataset"""
        return self.create_examples(os.path.join(data_dir, 'D-set-test.arin'), 
                                    os.path.join(data_dir, 'D-set-test.ar.M'))

class Vocabulary:
    """Base vocabulary class"""
    def __init__(self, token_to_idx=None):
        
        if token_to_idx is None:
            token_to_idx = dict()
        
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
    
    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index
    
    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]
    
    def lookup_token(self, token):
        return self.token_to_idx[token]
    
    def lookup_index(self, index):
        return self.idx_to_token[index]
    
    def to_serializable(self):
        return {'token_to_idx': self.token_to_idx}
    
    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)
    
    def __len__(self):
        return len(self.token_to_idx)
    
class SeqVocabulary(Vocabulary):
    """Sequence vocabulary class"""
    def __init__(self, token_to_idx=None, unk_token='<unk>',
                 pad_token='<pad>', sos_token='<s>',
                 eos_token='</s>'):
        
        super(SeqVocabulary, self).__init__(token_to_idx)
        
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        
        self.pad_idx = self.add_token(self.pad_token)
        self.unk_idx = self.add_token(self.unk_token)
        self.sos_idx = self.add_token(self.sos_token)
        self.eos_idx = self.add_token(self.eos_token)
        
    def to_serializable(self):
        contents = super(SeqVocabulary, self).to_serializable()
        contents.update({'unk_token': self.unk_token,
                         'pad_token': self.pad_token,
                         'sos_token': self.sos_token, 
                         'eos_token': self.eos_token})
        return contents
    
    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)
    
    def lookup_token(self, token):
        return self.token_to_idx.get(token, self.unk_idx)