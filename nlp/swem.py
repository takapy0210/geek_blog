import numpy as np
import pandas as pd
from gensim.models import word2vec


class SWEM():
    """単語埋め込み (Word Embedding) のみを利用して文章埋め込み (Sentence Embedding) を計算する

    参考URL:https://arxiv.org/abs/1805.09843v1

    Attributes:
        word2vec (word2vec): word2vecの事前学習モデル
        dim (int): word2vecの事前学習モデルの次元数
        oov_initialize_range (int): word2vecの事前学習モデルに含まれていない単語に割り当てるベクトル

    """

    def __init__(self, word2vec_model_name):
        self.word2vec = word2vec.Word2Vec.load(word2vec_model_name)
        self.dim = self.word2vec.trainables.layer1_size
        self.oov_initialize_range = (-0.01, 0.01)

    def get_word_embeddings(self, words) -> list:
        """word2vecから単語のベクトルを取得

        Args:
            words (list of str): 重みを取得したい単語のリスト

        Returns:
            list (float): 全単語のベクトルが格納された2次元リスト

        """
        np.random.seed(abs(hash(len(words))) % (10 ** 8))
        vectors = []
        for w in words:
            if w in self.word2vec:
                vectors.append(self.word2vec[w])
            else:
                vectors.append(np.random.uniform(self.oov_initialize_range[0], self.oov_initialize_range[1], self.dim))
        return vectors

    def average_pooling(self, text) -> np.array:
        """textに含まれる全単語ベクトルの次元毎の平均を計算する

        Args:
            text (str): ベクトルを計算したい文章

        Returns:
            np.array: 計算後のベクトル

        """
        emb = []
        for words in text:
            word_embeddings = self.get_word_embeddings(words)
            emb.append(np.nanmean(word_embeddings, axis=0))
        return np.array(emb)

    def max_pooling(self, text) -> np.array:
        """textに含まれる全単語ベクトルの次元毎の最大値を計算する

        Args:
            text (str): ベクトルを計算したい文章

        Returns:
            np.array: 計算後のベクトル

        """
        emb = []
        for words in text:
            word_embeddings = self.get_word_embeddings(words)
            emb.append(np.max(word_embeddings, axis=0))
        return np.array(emb)

    def concat_average_max_pooling(self, text) -> np.array:
        """textに含まれる全単語ベクトルの次元毎の平均値と最大値を計算した後それぞれを結合したベクトルを計算する

        平均ベクトル[1, 3, 4, 2, -2]と最大値ベクトル[5, 7, 3, 1, 3]があった場合に
        [1, 3, 4, 2, -2, 5, 7, 3, 1, 3]のベクトルを定義しreturnする

        Args:
            text (str): ベクトルを計算したい文章

        Returns:
            np.array: 計算後のベクトル

        """
        emb = []
        for words in text:
            word_embeddings = self.get_word_embeddings(words)
            emb.append(np.r_[np.nanmean(word_embeddings, axis=0), np.max(word_embeddings, axis=0)])
        return np.array(emb)

    def hier_or_avg_pooling(self, text, window) -> np.array:
        """textに含まれる単語に対してn-gramのように固定長のウィンドウでaverage-poolingした結果に対してmax poolingする

        単語数がwindowに満たない場合は、単純な平均（average_pooling）を計算する

        Args:
            text (str): ベクトルを計算したい文章
            window (int): n-gramのウィンドウの幅

        Returns:
            np.array: 計算後のベクトル

        """
        emb = []
        for words in text:
            word_embeddings = self.get_word_embeddings(words)
            text_len = len(word_embeddings)
            if window > text_len:
                emb.append(np.nanmean(word_embeddings, axis=0))
            else:
                window_average_pooling_vec = [np.nanmean(word_embeddings[i:i + window], axis=0)
                                              for i in range(text_len - window + 1)]
                emb.append(np.max(window_average_pooling_vec, axis=0))
        return np.array(emb)

    def calculate_emb(self, df, col, window, swem_type) -> pd.DataFrame:
        """swemを用いて質問の埋め込みを算出する

        Args:
            df (pd.Dataframe): 対象のDF
            col (str): token化後のテキストが設定されているカラム名
            window (int): hierarchical_poolingする際のwindow数
            swem_type (int): SWEMをどの計算方法で算出するかを指定
                            （1:average_pooling, 2:max_pooling, 3:concat_average_max_pooling, 4:hier_or_avg_pooling）

        Returns:
            pd.DataFrame: 埋め込み(N次元)のデータフレーム

        """

        # 質問の埋め込みを計算
        # swem_typeによって埋め込みの計算処理を分ける
        if swem_type == 1:
            swem_emb = self.average_pooling(df[col].values.tolist())
        elif swem_type == 2:
            swem_emb = self.max_pooling(df[col].values.tolist())
        elif swem_type == 3:
            swem_emb = self.concat_average_max_pooling(df[col].values.tolist())
        else:
            swem_emb = self.hier_or_avg_pooling(df[col].values.tolist(), window)

        # データフレームに変換
        swem_emb = pd.DataFrame(swem_emb)
        swem_emb = swem_emb.add_prefix('d_')
        return swem_emb
