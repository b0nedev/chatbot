from konlpy.tag import Mecab
import tensorflow as tf
import numpy as np


class PreProcessing():
    def __init__(self, filename):
        self.mecab = Mecab()
        self.chat_json = list()
        self.vocaes = None
        self.comments = list()
        self.data_init(filename)
        self.vocaes_len = len(self.vocaes)
 
    def data_init(self, filename):
        self.csv2json(filename)
        self.load_words()

    def rm_dqt(self, comment):
        md_comm = list()
        for ch in comment:
            if ch != '"':
                md_comm.append(ch)
        return ''.join(md_comm)

    def csv2json(self, file_name):
        with open(file_name) as f:
            for i, line in enumerate(f):
                seperator = line.split(':')
                self.chat_json.append({
                    'name': self.rm_dqt(seperator[0].strip()),
                    'comment': self.rm_dqt(seperator[1].strip())})

    def load_words(self):
        import re
        voca_list = []
        for i, comm in enumerate(self.chat_json):
            comment = comm.get('comment')
            if comment in ['?', '!', ',', '.', ':', ';']:
                voca_list.append(comment[-1])
                comment = comment[:-1]
            voca_list.extend(([w for w in comment.split(' ')]))
        self.vocaes = {w:i for i, w in enumerate(['_PAD_', '_GO_', '_EOS_', '_NEW_'])}
        self.vocaes.update({w:i+3 for i, w in enumerate(list(set(voca_list)))})

    def load_comments(self):
        for i, comm in enumerate(self.chat_json):
            self.comments.append([self.vocaes[w] for w in comm.get('comment').split(' ')])

    def max_len_stc(self):
        max_len_q, max_len_r  = 0, 0
        self.load_comments()
        for i, stc in enumerate(self.comments):
            if i % 2 == 0:
                len_query = len(stc)
                if len_query > max_len_q:
                    max_len_q = len_query
            else:
                len_response = len(stc)
                if len_response > max_len_r:
                    max_len_r = len_response
        return max_len_q, max_len_r + 1 
            
    def div2qr(self, comments):
        q, r = list(), list()
        for i in range(0, len(comments)-1, 2):
            q.append(comments[i])
            r.append(comments[i + 1])
        return q, r

    def _padding(self, seq, max_len, go=None, eos=None):
        if go:
            padded_seq = [self.vocaes['_GO_']] + seq
        elif eos:
            padded_seq = seq + [self.vocaes['_EOS_']]
        else:
            padded_seq = seq

        if len(padded_seq) < max_len:
            padded_seq = padded_seq + [self.vocaes['_PAD_']] * (max_len - len(padded_seq))

        return padded_seq
    
    def list2mats(self):
        en_input, de_input, targets = [],[],[]
        max_len_q, max_len_r = self.max_len_stc()
        qs, rs = self.div2qr(self.comments)
        for q in qs:
            en_input.append(np.eye(self.vocaes_len)[self._padding(q, max_len_q)])
        for r in rs:
            de_input.append(np.eye(self.vocaes_len)[self._padding(r, max_len_r, go=True)])
            targets.append(self._padding(r, max_len_r, eos=True))
        return en_input, de_input, targets
    
    def i2w(self, idx):
        voca_list = [w for w, i in self.vocaes.items()]
        return voca_list[idx]

    def add_ne(self, comm):
        word_list = []
        for word in comm.split(' '):
            if word not in self.vocaes:
                word = '_NEW_'
            word_list.append(word)
        return word_list
    
    def comm2vec(self, comm):
        word_list = self.add_ne(comm)
        return [self.vocaes[w] for w in word_list]
    
    def vec2comm(self, vec):
        comm_list = []
        for js in self.chat_json:
            commvec = self.comm2vec(js['comment'])
            comm_list.append({'commvec': commvec,'comment': js['comment']})

        for comm in comm_list:
            if comm['commvec'] == vec:
                return comm['comment']
    
    def list2mats2(self, q, r):
        en_input, de_input = [],[]
        max_len_q, max_len_r = self.max_len_stc()
        en_input.append(np.eye(self.vocaes_len)[self._padding(q, max_len_q)])
        de_input.append(np.eye(self.vocaes_len)[self._padding(r, max_len_r, go=True)])
        return en_input, de_input