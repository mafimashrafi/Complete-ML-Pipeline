import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')

#Ensure the logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

#setting logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text ]

    return " ".join(text)

def preprocess_df(df, text_columns = 'text', target_column='target'):
    try:
        logger.debug('Starting preprocessing for DataFrame')
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])

        logger.debug('Target column encoded')

        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        df.loc[:, text_columns] = df[text_columns].apply(transform_text)
        logger.debug('Text columns transformed')
        return df
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_columns = 'text', target_column = 'target'):
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        train_processed_data = preprocess_df(train_data, text_columns, target_column)
        test_processed_data = preprocess_df(test_data, text_columns, target_column)

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('Empty dataframe: %s', e)
    except Exception as e:
        logger.error('Error during data preprocessing: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()