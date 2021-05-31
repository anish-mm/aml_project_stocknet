import numpy as np
import torch
import os
import os.path as osp
from datetime import datetime, timedelta
import json
import random
import tqdm

class StocknetDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        start_date, end_date,
        vocab, vocab_size,
        stock_symbols,
        movement_path, tweet_path,
        word_embed_size,
        word_embed_type,
        glove_path,
        max_n_days,
        max_n_msgs,
        max_n_words,
        y_size,
        shuffle=True,
    ):
        super(StocknetDataset, self).__init__()

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        self.start_date = start_date
        self.end_date = end_date
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.stock_symbols = stock_symbols
        self.movement_path = movement_path
        self.tweet_path = tweet_path
        self.shuffle = shuffle
        self.word_embed_size=word_embed_size
        self.word_embed_type = word_embed_type
        self.glove_path = glove_path
        self.max_n_days = max_n_days
        self.max_n_msgs = max_n_msgs
        self.max_n_words = max_n_words
        self.y_size=y_size

        print(f"starting date = {self.start_date}")
        print(f"end date = {self.end_date}")

        # create the dictionaries, lists, etc.
        self.vocab_id_dict = self.index_token(vocab, key='token')
        self.stock_id_dict = self.index_token(stock_symbols, key='token', type='stock')
        self.load_all_technical()
        # self.load_all_tweets()
        self.n_samples = self.length()

    def _iterator(self):
        generators = [self.sample_gen_from_one_stock(s) for s in self.stock_symbols]
        
        while True:
            gen_id = random.randint(0, len(generators)-1)
            try:
                sample_dict = next(generators[gen_id])
                yield sample_dict
            except StopIteration:
                del generators[gen_id]
                if generators:
                    continue
                else:
                    # raise StopIteration
                    return

    def __iter__(self):
        return self._iterator()

    def length(self):
        tech_dict = self.tech_dict
        num = 0
        for s in tech_dict:
            num += len(tech_dict[s])
        return num

    def init_word_table(self):
        word_table_init = np.random.random((self.vocab_size, self.word_embed_size)) * 2 - 1  # [-1.0, 1.0]

        if self.word_embed_type != 'rand':
            n_replacement = 0
            # vocab_id_dict = self.index_token(vocab, key='token')
            vocab_id_dict = self.vocab_id_dict

            with open(self.glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tuples = line.split()
                    word, embed = tuples[0], [float(embed_col) for embed_col in tuples[1:]]
                    if word in ['<unk>', 'unk']:  # unify UNK
                        word = 'UNK'
                    if word in vocab_id_dict:
                        n_replacement += 1
                        word_id = vocab_id_dict[word]
                        word_table_init[word_id] = embed
        return word_table_init

    def load_all_technical(self):
        print("loading all technical data...")
        end_date = self.end_date
        start_date = self.start_date - timedelta(days=self.max_n_days)
        print(f"starting from {start_date}, to {end_date}")

        tech_dict = dict()
        for s in tqdm.tqdm(self.stock_symbols):
            s_dict = dict()
            path = osp.join(self.movement_path, f"{s}.txt")
            with open(path, "r", encoding='utf8') as fin:
                for line in fin:
                    data = line.split('\t')
                    t = datetime.strptime(data[0], '%Y-%m-%d').date()
                    if t > end_date:
                        continue
                    if t < start_date:
                        break # these files are sorted in descending order of date.

                    s_dict[t] = data[1:]
            tech_dict[s] = s_dict
        self.tech_dict = tech_dict
    
    def load_all_tweets(self):
        print("loading all tweets...")
        end_date = self.end_date
        start_date = self.start_date - timedelta(days=self.max_n_days)

        tweet_dict = dict()
        for s in tqdm.tqdm(self.stock_symbols):
            s_dict = dict()
            stock_tweet_path = osp.join(self.tweet_path, s)

            d = end_date
            while d >= start_date:
                msg_fp = os.path.join(stock_tweet_path, d.isoformat())
                if osp.exists(msg_fp):
                    d_list = list()
                    with open(msg_fp, 'r') as tweet_f:
                        for line in tweet_f:
                            text = json.loads(line)["text"]
                            d_list.append(text)
                    s_dict[d] = d_list
                d -= timedelta(days=1)
            tweet_dict[s] = s_dict
        self.tweet_dict = tweet_dict

    @staticmethod
    def _convert_token_to_id(token, token_id_dict):
        if token not in token_id_dict:
            token = 'UNK'
        return token_id_dict[token]
    
    def index_token(self, token_list, key='id', type='word'):
        """
        Create dictionaries mapping index to token or token to index.
        """
        assert key in ('id', 'token')
        assert type in ('word', 'stock')
        indexed_token_dict = dict()

        if type == 'word':
            token_list_cp = list(token_list)  # un-change the original input
            token_list_cp.insert(0, 'UNK')  # for unknown tokens
        else:
            token_list_cp = token_list

        if key == 'id':
            for id in range(len(token_list_cp)):
                indexed_token_dict[id] = token_list_cp[id]
        else:
            for id in range(len(token_list_cp)):
                indexed_token_dict[token_list_cp[id]] = id

        # id_token_dict = dict(zip(token_id_dict.values(), token_id_dict.keys()))
        return indexed_token_dict
    
    def _convert_words_to_ids(self, words, vocab_id_dict):
        """
            Replace each word in the tweet with its index in the dictionary
        """
        return [self._convert_token_to_id(w, vocab_id_dict) for w in words]

    def _get_prices_and_ts(self, ss, main_target_date):

        def _get_mv_class(data, use_one_hot=False):
            mv = float(data[0])
            if self.y_size == 2:
                if mv <= 1e-7:
                    return [1.0, 0.0] if use_one_hot else 0
                else:
                    return [0.0, 1.0] if use_one_hot else 1

            if self.y_size == 3:
                threshold_1, threshold_2 = -0.004, 0.005
                if mv < threshold_1:
                    return [1.0, 0.0, 0.0] if use_one_hot else 0
                elif mv < threshold_2:
                    return [0.0, 1.0, 0.0] if use_one_hot else 1
                else:
                    return [0.0, 0.0, 1.0] if use_one_hot else 2

        def _get_y(data):
            return _get_mv_class(data, use_one_hot=True)

        def _get_prices(data):
            return [float(p) for p in data[2:5]]

        def _get_mv_percents(data):
            return _get_mv_class(data)

        ts, ys, prices, mv_percents, main_mv_percent = list(), list(), list(), list(), 0.0
        # this takes care of the lag window.
        d_t_min = main_target_date - timedelta(days=self.max_n_days-1)

        s_dict = self.tech_dict[ss]
        d = main_target_date
        data = s_dict[d]
        ts.append(d)
        ys.append(_get_y(data))
        main_mv_percent = data[0]
        if -0.005 <= float(main_mv_percent) < 0.0055:  # discard sample with low movement percent
            return None
        d -= timedelta(days=1)
        while d >= d_t_min:
            if d in s_dict:
                data = s_dict[d]
                ts.append(d)
                ys.append(_get_y(data))
                prices.append(_get_prices(data))  # high, low, close
                mv_percents.append(_get_mv_percents(data))
            d -= timedelta(days=1)
        try:
            while 1:
                if d in s_dict:
                    data = s_dict[d]
                    prices.append(_get_prices(data))
                    mv_percents.append(_get_mv_percents(data))
                    break
                d -= timedelta(days=1)
        except:
            print(f"main target = {main_target_date}")
            print(f"d_t_min = {d_t_min}")
            print(f"d = {d}")
        
        T = len(ts)
        if len(ys) != T or len(prices) != T or len(mv_percents) != T:  # ensure data legibility
            return None

        # ascend
        for item in (ts, ys, mv_percents, prices):
            item.reverse()

        prices_and_ts = {
            'T': T,
            'ts': ts,
            'ys': ys,
            'main_mv_percent': main_mv_percent,
            'mv_percents': mv_percents,
            'prices': prices,
        }

        return prices_and_ts

    def _get_unaligned_corpora(self, ss, main_target_date,):
        def get_ss_index(word_seq, ss):
            ss = ss.lower()
            ss_index = len(word_seq) - 1  # init
            if ss in word_seq:
                ss_index = word_seq.index(ss)
            else:
                if '$' in word_seq:
                    dollar_index = word_seq.index('$')
                    if dollar_index is not len(word_seq) - 1 and ss in word_seq[dollar_index + 1]:
                        ss_index = dollar_index + 1
                    else:
                        for index in range(dollar_index + 1, len(word_seq)):
                            if ss in word_seq[index]:
                                ss_index = index
                                break
            return ss_index

        unaligned_corpora = list()  # list of sets: (d, msgs, ss_indices)
        stock_tweet_path = os.path.join(str(self.tweet_path), ss)

        d_d_max = main_target_date - timedelta(days=1)
        d_d_min = main_target_date - timedelta(days=self.max_n_days)

        # s_dict = self.tweet_dict[ss]
        d = d_d_max  # descend
        while d >= d_d_min:
            msg_fp = os.path.join(stock_tweet_path, d.isoformat())
            if os.path.exists(msg_fp):
                word_mat = np.zeros([self.max_n_msgs, self.max_n_words], dtype=np.int32)
                n_word_vec = np.zeros([self.max_n_msgs, ], dtype=np.int32)
                ss_index_vec = np.zeros([self.max_n_msgs, ], dtype=np.int32)
                msg_id = 0
                with open(msg_fp, 'r') as tweet_f:
                    for line in tweet_f:
                        msg_dict = json.loads(line)
                        text = msg_dict['text']
                        if not text:
                            continue

                        words = text[:self.max_n_words]
                        word_ids = self._convert_words_to_ids(words, self.vocab_id_dict)
                        n_words = len(word_ids)

                        n_word_vec[msg_id] = n_words
                        word_mat[msg_id, :n_words] = word_ids
                        ss_index_vec[msg_id] = get_ss_index(words, ss)

                        msg_id += 1
                        if msg_id == self.max_n_msgs:
                            break
                corpus = [d, word_mat[:msg_id], ss_index_vec[:msg_id], n_word_vec[:msg_id], msg_id]
                unaligned_corpora.append(corpus)
            d -= timedelta(days=1)

        unaligned_corpora.reverse()  # ascend
        return unaligned_corpora

    def _trading_day_alignment(self, ts, T, unaligned_corpora):
        aligned_word_tensor = np.zeros([T, self.max_n_msgs, self.max_n_words], dtype=np.int32)
        aligned_ss_index_mat = np.zeros([T, self.max_n_msgs], dtype=np.int32)
        aligned_n_words_mat = np.zeros([T, self.max_n_msgs], dtype=np.int32)
        aligned_n_msgs_vec = np.zeros([T, ], dtype=np.int32)

        # list for gathering
        aligned_msgs = [[] for _ in range(T)]
        aligned_ss_indices = [[] for _ in range(T)]
        aligned_n_words = [[] for _ in range(T)]
        aligned_n_msgs = [[] for _ in range(T)]

        corpus_t_indices = []
        max_threshold = 0

        for corpus in unaligned_corpora:
            d = corpus[0]
            for t in range(T):
                if d < ts[t]:
                    corpus_t_indices.append(t)
                    break

        assert len(corpus_t_indices) == len(unaligned_corpora)

        for i in range(len(unaligned_corpora)):
            corpus, t = unaligned_corpora[i], corpus_t_indices[i]
            word_mat, ss_index_vec, n_word_vec, n_msgs = corpus[1:]
            aligned_msgs[t].extend(word_mat)
            aligned_ss_indices[t].extend(ss_index_vec)
            aligned_n_words[t].append(n_word_vec)
            aligned_n_msgs[t].append(n_msgs)

        def is_eligible():
            n_fails = len([0 for n_msgs in aligned_n_msgs if sum(n_msgs) == 0])
            return n_fails <= max_threshold

        if not is_eligible():
            return None

        # gather into nd_array and clip exceeded part
        for t in range(T):
            n_msgs = sum(aligned_n_msgs[t])

            if aligned_msgs[t] and aligned_ss_indices[t] and aligned_n_words[t]:
                msgs, ss_indices, n_word = np.vstack(aligned_msgs[t]), np.hstack(aligned_ss_indices[t]), np.hstack(aligned_n_words[t])
                assert len(msgs) == len(ss_indices) == len(n_word)
                n_msgs = min(n_msgs, self.max_n_msgs)  # clip length
                aligned_n_msgs_vec[t] = n_msgs
                aligned_word_tensor[t, :n_msgs] = msgs[:n_msgs]
                aligned_ss_index_mat[t, :n_msgs] = ss_indices[:n_msgs]
                aligned_n_words_mat[t, :n_msgs] = n_word[:n_msgs]

        aligned_info_dict = {
            'msgs': aligned_word_tensor,
            'ss_indices': aligned_ss_index_mat,
            'n_words': aligned_n_words_mat,
            'n_msgs': aligned_n_msgs_vec,
        }

        return aligned_info_dict

    def sample_gen_from_one_stock(self, s):
        """
            generate samples for the given stock.

            => tuple, (x:dict, y_:int, price_seq: list of floats, prediction_date_str:str)
        """
        main_target_dates = []
        for d in self.tech_dict[s].keys():
            if self.start_date <= d <= self.end_date:
                main_target_dates.append(d)

        if self.shuffle:  # shuffle data
            random.shuffle(main_target_dates)
        #dict_nbrs: s -> s_nbr
        for main_target_date in main_target_dates:
            # logger.info('start _get_unaligned_corpora')
            unaligned_corpora = self._get_unaligned_corpora(s, main_target_date)
            # logger.info('start _get_prices_and_ts')
            prices_and_ts = self._get_prices_and_ts(s, main_target_date)
            # prices_and_ts = self._get_prices_and_ts(s_nbr, main_target_date)
            if not prices_and_ts:
                continue

            # logger.info('start _trading_day_alignment')
            aligned_info_dict = self._trading_day_alignment(prices_and_ts['ts'], prices_and_ts['T'], unaligned_corpora)
            if not aligned_info_dict:
                continue

            ys = np.zeros([self.max_n_days, self.y_size], dtype=np.float32)
            T = prices_and_ts['T']
            ys[:T] = prices_and_ts['ys']
            mv_percent = np.zeros([self.max_n_days], dtype=np.float32)
            mv_percent[:T] = prices_and_ts['mv_percents']
            # we would want to change this if we are doing herd-movement thingy.
            price = np.zeros([self.max_n_days, 3], dtype=np.float32)
            price[:T] = prices_and_ts['prices']
            word = np.zeros([self.max_n_days, self.max_n_msgs, self.max_n_words], dtype=np.int32)
            word[:T] = aligned_info_dict['msgs']
            # is this a mask for whether there is a message present?
            ss_index = np.zeros([self.max_n_days, self.max_n_msgs], dtype=np.int32)
            ss_index[:T] = aligned_info_dict['ss_indices']
            n_msgs = np.zeros([self.max_n_days], dtype=np.int64)
            n_msgs[:T] = aligned_info_dict['n_msgs']
            n_words = np.zeros([self.max_n_days, self.max_n_msgs], dtype=np.int32)
            n_words[:T] = aligned_info_dict['n_words']

            sample_dict = {
                # meta info
                'stock': self._convert_token_to_id(s, self.stock_id_dict),
                'main_target_date': main_target_date.isoformat(),
                'T': T,
                # target
                'ys': ys,
                'main_mv_percent': prices_and_ts['main_mv_percent'],
                'mv_percents': mv_percent,
                # source
                'prices': price,
                'msgs': word,
                'ss_indices': ss_index,
                'n_words': n_words,
                'n_msgs': n_msgs,
            }

            yield sample_dict

