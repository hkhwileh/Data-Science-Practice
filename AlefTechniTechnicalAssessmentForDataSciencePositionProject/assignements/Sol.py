import re
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

import tokenization

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_hub as hub
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

class DataLoader:
    def load(self, file_path):
        df = self.read(self,file_path)
        df = df[['Question', 'Answer', 'ReferenceAnswers', 'Annotation']]
        df['Annotation'] = df['Annotation'].apply(self.get_annotation)
        train, test, y_train, y_test = train_test_split(df, df['Annotation'], test_size=0.20, stratify=df['Annotation'],
                                                        random_state=777)
        train = self.process_data(train)
        original_test = test.copy()
        test = self.process_data(test)
        return train, y_train, test, y_test, original_test



    def process_data(self, dataframe):
        dataframe = self.clean_text(self.concat_text(self.expload_reference_answers(dataframe)))
        return dataframe

    def read(file_path):
        xml_data = open(file_path).read()
        root = ET.XML(xml_data)
        data = []
        for i, child in enumerate(root):
            row = {}
            for attribute in child:
                if attribute.tag == "ProblemDescription":
                    row[attribute.tag] = attribute.text
                elif attribute.tag == "Question":
                    row[attribute.tag] = attribute.text
                elif attribute.tag == "Answer":
                    row[attribute.tag] = attribute.text
                elif attribute.tag == "Annotation":
                    row[attribute.tag] = attribute.get("Label")
                    for subchild in attribute:
                        if subchild.tag == "AdditionalAnnotation":
                            row["ContextRequired"] = subchild.get("ContextRequired")
                            row["ExtraInfoInAnswer"] = subchild.get("ExtraInfoInAnswer")
                        elif subchild.tag == "Comments":
                            row[subchild.tag] = subchild.text
                            row["Watch"] = subchild.get("Watch")
                elif attribute.tag == "ReferenceAnswers":
                    row[attribute.tag] = attribute.text
            data.append(row)
        return pd.DataFrame(data)


def xml2df(xml_source, df_cols, source_is_file = False, show_progress=True):
    """Parse the input XML source and store the result in a pandas
    DataFrame with the given columns.

    """
    if source_is_file:
        xtree = ET.parse(xml_source) # xml_source = xml_file
        xroot = xtree.getroot()
    else:
        xroot = ET.fromstring(xml_source) # xml_source = xml_string
    consolidator_dict = dict()
    default_instance_dict = {label: None for label in df_cols}

    def get_children_info(children, instance_dict):
        # We avoid using element.getchildren() as it is deprecated.
        # Instead use list(element) to get a list of attributes.
        for child in children:

            if len(list(child))>0:
                instance_dict = get_children_info(list(child),
                                                  instance_dict)

            if len(list(child.keys()))>0:
                items = child.items()
                instance_dict.update({key: value for (key, value) in items})

            #print(child.keys())
            instance_dict.update({child.tag: child.text})
        return instance_dict

    # Loop over all instances
    for instance in list(xroot):
        instance_dict = default_instance_dict.copy()
        ikey, ivalue = instance.items()[0] # The first attribute is "ID"
        instance_dict.update({ikey: ivalue})
        if show_progress:
            print('{}: {}={}'.format(instance.tag, ikey, ivalue))
        # Loop inside every instance
        instance_dict = get_children_info(list(instance),
                                          instance_dict)

        #consolidator_dict.update({ivalue: instance_dict.copy()})
        consolidator_dict[ivalue] = instance_dict.copy()
    df = pd.DataFrame(consolidator_dict).T
    df = df[df_cols]

    return df

xml_source = r'C:\alef\grade_data.xml'
df_cols = ["ID", "TaskID", "DataSource", "ProblemDescription", "Question", "Answer",
           "ContextRequired", "ExtraInfoInAnswer", "Comments", "Watch", 'ReferenceAnswers']
df = xml2df(xml_source, df_cols, source_is_file = True)


def get_annotation(text):
    values = text.split('|')
    annotation = [i for i, value in enumerate(values) if "(1)" in value][0]
    return annotation


def split_reference_answers(text):
    return list(filter(None, re.split("\d+:  ", text.replace("\n", ""))))


def clean_text(df, column='text'):
    df[column] = df[column].str.replace('\n', ' ')
    df[column] = df[column].str.replace('\r', ' ')
    df[column] = df[column].str.replace('\t', ' ')

    # This removes unwanted texts
    df[column] = df[column].apply(lambda x: re.sub(r'[0-9]', ' ', x))
    df[column] = df[column].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]', ' ', x))

    # Converting all upper case to lower case
    df[column] = df[column].apply(lambda s: s.lower() if type(s) == str else s)

    # Remove un necessary white space
    df[column] = df[column].str.replace('  ', ' ')
    return df


def expload_reference_answers(self, dataframe):
    dataframe.ReferenceAnswers = dataframe.ReferenceAnswers.apply(self.split_reference_answers)
    dataframe = dataframe.explode('ReferenceAnswers')
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def concat_text(dataframe):
    dataframe['text'] = (dataframe['ReferenceAnswers'] + ' ' + dataframe['Answer'] + ' ' + dataframe['Question']).apply(
        lambda row: row.strip())
    return dataframe

data_loader = DataLoader()
train, y_train, test, y_test, original_test = data_loader.load('C:\alef\grade_data.xml')



class Model:
    def __init__(self, data_loader):
        self.max_len = 200
        self.class_dict = {0: 'correct', 1: 'correct_but_incomplete', 2: 'contradictory', 3: 'incorrect'}
        self.data_loader = data_loader
        module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
        self.bert_layer = hub.KerasLayer(module_url, trainable=True)
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        self.model = self.build_model()

    def get_class_weights(self, labels):
        weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(labels),
                                                    labels)
        class_weights = {}
        for w, c in zip(weights, np.unique(labels)):
            class_weights[c] = w
        return class_weights

    def bert_encode(self, texts):
        all_tokens = []
        all_masks = []
        all_segments = []

        for text in texts:
            text = self.tokenizer.tokenize(text)

            text = text[:self.max_len - 2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = self.max_len - len(input_sequence)

            tokens = self.tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * self.max_len

            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)

        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

    def encode_inputs(self, data):
        data_input = self.bert_encode(data.text.values)
        y_data_org = data.pop('Annotation')
        y_data = to_categorical(np.asarray(y_data_org))
        return data_input, y_data, y_data_org

    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def train(self, train_input):
        checkpoint = tf.keras.callbacks.ModelCheckpoint('new-dt-model.h5', monitor='val_accuracy', save_best_only=True,
                                                        verbose=1)
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, verbose=1)
        train_input, y_train, y_train_org = self.encode_inputs(train_input)
        class_weight = self.get_class_weights(y_train_org)
        print('class weights: ', class_weight)
        train_history = self.model.fit(
            train_input, y_train,
            validation_split=0.2,
            # class_weight=class_weight,
            epochs=50,
            callbacks=[checkpoint, earlystopping],
            batch_size=32,
            verbose=1
        )

    def build_model(self):
        input_word_ids = tf.keras.Input(shape=(self.max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(shape=(self.max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.Input(shape=(self.max_len,), dtype=tf.int32, name="segment_ids")

        _, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        net = tf.keras.layers.Dense(128, activation='relu')(clf_output)
        net = tf.keras.layers.Dropout(0.2)(net)
        net = tf.keras.layers.Dense(32, activation='relu')(net)
        net = tf.keras.layers.Dropout(0.2)(net)
        out = tf.keras.layers.Dense(4, activation='softmax')(net)

        model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy',
                      metrics=['accuracy', self.f1_m, self.precision_m, self.recall_m])
        print(model.summary())

        return model

    def load(self, model_path):
        self.model.load_weights(model_path)

    def test(self, test):
        t_labels = []
        p_labels = []
        for i, row in test.iterrows():
            p = self.vote_predict_df(row)
            t_labels.append(row.Annotation)
            p_labels.append(p)
        class_ids = list(np.unique(test.Annotation))
        class_names = [self.class_dict[cid] for cid in class_ids]
        clf_report = classification_report(t_labels, p_labels, target_names=class_names)
        print(clf_report)

    def vote_predict_df(self, row):
        row = pd.DataFrame({
            "Question": [row.Question],
            "Answer": [row.Answer],
            "ReferenceAnswers": [row.ReferenceAnswers]
        })
        row = self.data_loader.process_data(row)
        row_input = self.bert_encode(row.text.values)
        preds = self.model.predict(row_input)
        preds = np.argmax(preds, axis=1)
        return np.bincount(preds).argmax()

    def vote_predict(self, question, answer, reference_answers):
        row = pd.DataFrame({
            "Question": [question],
            "Answer": [answer],
            "ReferenceAnswers": [reference_answers]
        })
        row = self.data_loader.process_data(row)
        row_input = self.bert_encode(row.text.values)
        preds = self.model.predict(row_input)
        preds = np.argmax(preds, axis=1)
        return self.class_dict[np.bincount(preds).argmax()]

