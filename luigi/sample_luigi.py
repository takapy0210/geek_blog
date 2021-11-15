import pickle
import warnings
import logging

import pandas as pd
import luigi
from luigi.util import requires
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")
logger = logging.getLogger()


class LoadDataset(luigi.Task):
    """データセットをロードするクラス"""
    task_namespace = 'titanic_tasks'

    def output(self):
        # return luigi.LocalTarget("data/titanic.csv")  # csvで出力する場合
        return luigi.LocalTarget("data/titanic.pkl", format=luigi.format.Nop)

    def run(self):
        # titanicデータの読み込み
        df = datasets.fetch_openml("titanic", version=1, as_frame=True, return_X_y=False).frame
        logger.info(f'Data shape: {df.shape}')

        # pklで出力する
        with self.output().open('w') as f:
            f.write(pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL))

        # csvで出力したい場合は普通にpandasで出力する
        # 型が崩れる可能性があるので非推奨ではある
        # df.to_csv("data/titanic.csv", index=False)


@requires(LoadDataset)
class Processing(luigi.Task):
    """データの加工を行う"""
    task_namespace = 'titanic_tasks'

    def output(self):
        # return luigi.LocalTarget("data/processing_titanic.csv")  # csvで出力する場合
        return luigi.LocalTarget("data/processing_titanic.pkl", format=luigi.format.Nop)

    def run(self):
        # データの読み込み
        with self.input().open() as f:
            # df = pd.read_csv(f)  # pandasで読み込むパターン
            df = pickle.load(f)  # pickleで読み込むパターン
        logger.info(f'Before Data shape: {df.shape}')

        # 欠損値処理
        df.loc[:, 'age'] = df['age'].fillna(df['age'].mean())
        df.loc[:, 'fare'] = df['fare'].fillna(df['fare'].mean())

        # カテゴリエンコード
        categorical_cols = ["pclass", "sex", "embarked"]
        df = self.sklearn_oh_encoder(df=df, cols=categorical_cols, drop_col=True)
        logger.info(f'After Data shape: {df.shape}')

        # 学習に使用するカラムのみを出力
        use_cols = [
            'survived',
            'age',
            'sibsp',
            'parch',
            'fare',
            'pclass_1.0',
            'pclass_2.0',
            'pclass_3.0',
            'sex_female',
            'sex_male',
            'embarked_C',
            'embarked_Q',
            'embarked_S',
            'embarked_nan'
        ]
        df = df[use_cols]

        # 保存
        with self.output().open('w') as f:
            f.write(pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL))

    def sklearn_oh_encoder(self, df, cols, drop_col=False):
        """カテゴリ変換
        sklearnのOneHotEncoderでEncodingを行う

        Args:
            df: カテゴリ変換する対象のデータフレーム
            cols (list of str): カテゴリ変換する対象のカラムリスト
            drop_col (bool): エンコード対象のカラムを削除するか否か

        Returns:
            pd.Dataframe: dfにカテゴリ変換したカラムを追加したデータフレーム
        """
        output_df = df.copy()
        for col in cols:
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            ohe_df = pd.DataFrame((ohe.fit_transform(output_df[[col]])), columns=ohe.categories_[0])
            ohe_df = ohe_df.add_prefix(f'{col}_')
            # 元のDFに結合
            output_df = pd.concat([output_df, ohe_df], axis=1)
            if drop_col:
                output_df = output_df.drop(col, axis=1)
        return output_df


@requires(Processing)
class TrainTestSplit(luigi.Task):
    """データを学習データと検証データに分割する"""
    task_namespace = 'titanic_tasks'

    def output(self):
        return [luigi.LocalTarget("data/processing_titanic_train.pkl", format=luigi.format.Nop),
                luigi.LocalTarget("data/processing_titanic_test.pkl", format=luigi.format.Nop)]

    def run(self):
        # データの読み込み
        with self.input().open() as f:
            df = pickle.load(f)  # pickleで読み込むパターン

        train, test = train_test_split(df, test_size=0.3, shuffle=True, stratify=df['survived'], random_state=42)
        logger.info(f'Train shape: {train.shape}')
        logger.info(f'Test shape: {test.shape}')

        with self.output()[0].open('w') as f:
            f.write(pickle.dumps(train, protocol=pickle.HIGHEST_PROTOCOL))

        with self.output()[1].open('w') as f:
            f.write(pickle.dumps(test, protocol=pickle.HIGHEST_PROTOCOL))


@requires(TrainTestSplit)
class Training(luigi.Task):
    """学習"""
    task_namespace = 'titanic_tasks'

    def output(self):
        return luigi.LocalTarget("model/random_forest.model", format=luigi.format.Nop)

    def run(self):
        # データの読み込み
        with self.input()[0].open() as f:
            train = pickle.load(f)

        logger.info(f'Train shape: {train.shape}')

        target_col = 'survived'
        X_train = train.drop(target_col, axis=1)
        y_train = train[target_col]

        model = RandomForestClassifier(random_state=1)
        model.fit(X_train, y_train)

        # 保存
        with self.output().open('w') as f:
            f.write(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


@requires(TrainTestSplit, Training)
class Predict(luigi.Task):
    """推論"""
    task_namespace = 'titanic_tasks'

    def output(self):
        return luigi.LocalTarget("data/predict_data.csv")

    def run(self):
        # データの読み込み
        with self.input()[0][1].open() as f:
            valid = pickle.load(f)

        # モデルの読み込み
        with self.input()[1].open() as f:
            model = pickle.load(f)

        logger.info(f'Valid data shape: {valid.shape}')

        target_col = 'survived'
        X_valid = valid.drop(target_col, axis=1)
        y_valid = valid[target_col]

        # 予測
        y_pred = model.predict(X_valid)
        logger.info(f'Accuracy Score: {accuracy_score(y_valid, y_pred)}')
        logger.info('\n' + classification_report(y_valid, y_pred))

        # # 保存
        valid.loc[:, 'y_pred'] = y_pred
        valid.to_csv('data/predict_data.csv', index=False)


@requires(Predict)
class MyInvokerTask(luigi.WrapperTask):
    task_namespace = 'titanic_tasks'
    pass


if __name__ == '__main__':

    # 設定ファイルの読み込み
    luigi.configuration.LuigiConfigParser.add_config_path('./luigi.cfg')
    # 実行
    luigi.build([MyInvokerTask()], local_scheduler=True)
    # luigi.build([MyInvokerTask()], local_scheduler=False)  # ブラウザからチェックしたい場合はこちら
