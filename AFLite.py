from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class AFLite():
    def __init__(self, model, task, metric, value_metric, p_value, features, labels, validation_data, test_data, upper_bound, n_partition, train_size, predictability_treshold, slice_num) -> None:
        ''' Parametrs:
                model (): Модель, которая будет фильтровать данные.
                task (str): clf или NER. Задача, под которую фильтруются данные.
                metric (str): f1, precision и recall. Эта метрика дает оценку документу 0 или 1.  1 - удовлетворительный результат проогнозирования, 0 - нет.
                value_metric (float): Уровень метрики. Если ниже определенного уровня, то выставляется 0, иначе - 1
                p_value (float): Фильтрация уверенности модели.
                features (list): матрица признаков. Например Tf-idf.
                labels (list): Вектор меток.
                validation_data (list): проверочные данные для модели.
                test_data (list): Тестовые данные.
                upper_bound (int): Какое количество данных необходимо оставить.
                n_partition (int):
                train_size (float):
                predictability_treshold (float):  Пороговая оценка.
                slice_num (int): Критическая масса для удаления данных.
        '''

        self.model = model
        self.task = task
        self.metric = metric
        self.value_metric = value_metric
        self.p_value = p_value
        self.features = features
        self.labels = labels
        self.validation_data = validation_data
        self.test_data = test_data
        self.upper_bound = upper_bound
        self.n_partition = n_partition
        self.train_size = train_size
        self.predictability_treshold = predictability_treshold
        self.slice_num = slice_num

    def __calculate_right_clf(self, fact_class, predict_class) -> int:
        ''' Метод возвращает 1, если доля правильных классов выше установленного уровня, и 0, если меньше.
        '''

        if self.metric == 'f1':
            estimation = self.__f_1(fact_class, predict_class)
        elif self.metric == 'precision':
            estimation = self.__precision(fact_class, predict_class)
        elif self.metric == 'recall':
            estimation = self.__recall(fact_class, predict_class)

        if estimation < self.value_metric:
            return 0
        else:
            return 1

    def __f_1(self, fact_class, predict_class):
        '''
        '''
        rec = self.__recall(fact_class, predict_class)
        prec = self.__precision(fact_class, predict_class)
        f1 = (2*prec*rec) / (rec + prec)
        return f1
    
    def __precision(self, fact_class, predict_class):
        '''
        '''

        tp = []
        fp = []

        for index, val in enumerate(predict_class):
            if val == 1 and fact_class[index] == 1:
                tp.append(1)
            elif val == 1 and fact_class[index] == 0:
                fp.append(1)
        
        precision = len(tp) / (len(tp) + len(fp))
        return precision
    
    def __recall(self, fact_class, predict_class):
        '''
        '''

        tp = []
        fn = []
        for index, val in enumerate(predict_class):
            if val == 1 and fact_class[index] == 1:
                tp.append(1)
            elif val ==0 and fact_class[index] == 1:
                fn.append(1)

        recall = len(tp) / (len(tp) + len(fn))
        return recall

    def aflite_clf(self):
        '''
        '''

        while len(self.features.index) > self.upper_bound:

            acc = {}
            for _ in tqdm(range(self.n_partition)):
                X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, train_size=self.train_size)

                vectorizer = TfidfVectorizer()

                X_train = vectorizer.fit_transform(X_train.values.flatten().tolist())
                X_train = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())

                X_test = vectorizer.transform(X_test.values.flatten().tolist())
                X_test = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out(), index=y_test.index)
                self.model.fit(X_train, y_train)

                for index, feature in X_test.iterrows():
                    if index not in acc.keys():
                        acc[index] = []
                    pred = self.model.predict([feature.values.tolist()])
                    pred = pred.flatten()
                    pred = pred.tolist()
                    fact = y_test.loc[index]
                    fact = fact.values
                    fact = fact.tolist()
                        # return [index, X_test, y_test, pred]
                    estimation = self.__calculate_right_clf(fact, pred)
                    acc[index].append(estimation)

            removing_list = []
            for idx, predictions in acc.item():
                if len(predictions) >= 4:
                    predictability_score = sum(predictions) / len(predictions)
                    if predictability_score > self.predictability_treshold:
                        removing_list.append(idx)

            if len(removing_list) > self.slice_num:
                self.features = self.features.drop(index=removing_list)
                self.labels = self.labels.drop(index=removing_list)
            else:
                print('Недостаточно количества эксземпляров для удаления.')
                break
            
        return self.features, self.labels
