
import os
from time import time

import pandas as pd
import xml.etree.ElementTree as ET
from numpy.linalg import norm
from sklearn.model_selection import train_test_split

from Assignment.utils import get_hash
from Assignment.utils import get_reference_answers


def xml2df(xml_source, df_cols, source_is_file = False, show_progress=True):
    """Parse the input XML source and store the result in a pandas
    DataFrame with the given columns.

    For xml_source = xml_file, Set: source_is_file = True
    For xml_source = xml_string, Set: source_is_file = False

    <element attribute_key1=attribute_value1, attribute_key2=attribute_value2>
        <child1>Child 1 Text</child1>
        <child2>Child 2 Text</child2>
        <child3>Child 3 Text</child3>
    </element>
    Note that for an xml structure as shown above, the attribute information of
    element tag can be accessed by list(element). Any text associated with <element> tag can be accessed
    as element.text and the name of the tag itself can be accessed with
    element.tag.
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
            #print(child)
            #print(child.tag)
            #print(child.items())
            #print(child.getchildren()) # deprecated method
            #print(list(child))
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



def train_test_landmark_split(data):

    # Modeling/Test split
    remaining, test = train_test_split(data, test_size=0.2, random_state=22)
    # Train/Dev split
    train, landmarks = train_test_split(
        remaining,
        test_size=0.125,
        random_state=22)
    # Save modeling and test data
    if not os.path.exists('alef'):
        os.makedirs('alef')
    data.to_csv(os.path.join('alef', 'grade_data.csv'), index=False)
    train.to_csv(os.path.join('alef', 'train.csv'), index=False)
    landmarks.to_csv(os.path.join('alef', 'landmarks.csv'), index=False)
    test.to_csv(os.path.join('alef', 'test.csv'), index=False)
    print("INFO: Train, test and landmarks data were created successfully "
    "in alef directory.")


def landmarks(reference_landmarks, student_landmarks):
    """Create a text file with word2vec embeddings of student
    and reference answers and their labels.
    Args:
        reference_landmarks: csv to extract reference landmarks
        student_landmarks: csv to extract student landmarks
    """

    # Init timer
    start = time()

    # Read a dataset containing all reference answers
    ra_data = pd.read_csv(reference_landmarks)
    # Create hash keys for problem description and question
    ra_data['pd_hash'] = ra_data['problem_description'].apply(get_hash)
    ra_data['qu_hash'] = ra_data['question'].apply(get_hash)
    # Create a dataframe of reference answers one per row
    ra_data['ra_list'] = ra_data['reference_answers']\
        .apply(get_reference_answers)
    landmarks_ra = ra_data[['pd_hash', 'qu_hash', 'label', 'ra_list']]
    landmarks_ra = landmarks_ra.explode('ra_list')
    landmarks_ra = landmarks_ra.rename(columns={'ra_list':'answer'})
    landmarks_ra['label'] = 0 # these are possible correct answers (class 0)
    landmarks_ra = landmarks_ra.drop_duplicates()
    print("INFO: Found {} distinct reference landmark answers."\
        .format(len(landmarks_ra)))

    # Create a dataframe of student answers
    sa_data = pd.read_csv(student_landmarks)
    # Create hash keys for problem description and question
    sa_data['pd_hash'] = sa_data['problem_description'].apply(get_hash)
    sa_data['qu_hash'] = sa_data['question'].apply(get_hash)
    landmarks_sa = sa_data[['pd_hash', 'qu_hash', 'label', 'answer']]
    landmarks_sa = landmarks_sa.drop_duplicates()
    print("INFO: Found {} distinct student landmark answers."\
        .format(len(landmarks_sa)))

    # Create the landmarks dataframe with distinct answers and their labels
    landmarks = landmarks_ra.append(landmarks_sa).drop_duplicates()

    # Create a featurizer object that converts a phrase into embedding vector
    emb_file = os.path.join('data', 'GoogleNews-vectors-negative300.bin')
    featurizer = Featurizer(emb_file)

    # Save the embeddings and labels to disk
    n_landmarks = 0
    with open(os.path.join('alef', 'landmarks.txt'), 'w') as f:
        f.write(r'C:\alef')
        for i in range(len(landmarks)):
            pd_hash = landmarks.iloc[i]['pd_hash']
            qu_hash = landmarks.iloc[i]['qu_hash']
            label = landmarks.iloc[i]['label']
            answer = landmarks.iloc[i]['answer']
            emb = featurizer.doc2vec(landmarks.iloc[i]['answer'])
            emb_txt = ','.join(map(str, emb))
            if norm(emb) != 0:
                n_landmarks += 1
                f.write("%s\t%s\t%s\t%s\t%s\n"\
                    %(pd_hash, qu_hash, label, answer, emb_txt))
    print('INFO: Generating landmark embeddings took %.2f seconds.' \
        %(time() - start))
    print("INFO: Found {} non zero landmarks in total.".format(n_landmarks))

df = xml2df(xml_source, df_cols, source_is_file = True)
train_test_landmark_split(df)
landmarks(os.path.join('alef',  'grade_data.csv'),
        os.path.join('alef',  'landmarks.csv'))