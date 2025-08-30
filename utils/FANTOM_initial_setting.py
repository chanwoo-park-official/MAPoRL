import datetime
import hashlib
import os
import tarfile
import zipfile

import pandas as pd
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt


class DownloadableFile:
    def __init__(self, url, filename, expected_hash, version="1.0", zipped=True):
        self.url = url
        self.filename = filename
        self.expected_hash = expected_hash
        self.zipped = zipped
        self.version = version


FANTOM = DownloadableFile(
    url='https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz',
    filename='fantom.tar.gz',
    expected_hash='1d08dfa0ea474c7f83b9bc7e3a7b466eab25194043489dd618b4c5223e1253a4',
    version="1.0",
    zipped=True
)


# =============================================================================================================

def unzip_file(file_path, directory='.'):
    if file_path.endswith(".zip"):
        target_location = os.path.join(directory, os.path.splitext(os.path.basename(file_path))[0])
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(target_location)
    elif file_path.endswith(".tar.gz"):
        target_location = os.path.join(directory, os.path.basename(file_path).split(".")[0])
        with tarfile.open(file_path) as tar:
            tar.extractall(target_location)

    return target_location


def check_built(path, version_string=None):
    """
    Check if '.built' flag has been set for that task.
    If a version_string is provided, this has to match, or the version is regarded as not built.
    """
    fname = os.path.join(path, '.built')
    if not os.path.isfile(fname):
        return False
    else:
        with open(fname, 'r') as read:
            text = read.read().split('\n')
        return len(text) > 1 and text[1] == version_string


def mark_built(path, version_string="1.0"):
    """
    Mark this path as prebuilt.
    Marks the path as done by adding a '.built' file with the current timestamp plus a version description string.
    """
    with open(os.path.join(path, '.built'), 'w') as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write('\n' + version_string)


def download_and_check_hash(url, filename, expected_hash, version, directory='data', chunk_size=1024 * 1024 * 10):
    # Download the file
    response = requests.get(url, stream=True)
    try:
        total_size = int(response.headers.get('content-length', 0))
    except:
        print("Couldn't get content-length from response headers, using chunk_size instead")
        total_size = chunk_size
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    data = b''
    for chunk in response.iter_content(chunk_size=chunk_size):
        progress_bar.update(len(chunk))
        data += chunk
    progress_bar.close()

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the file to disk
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        f.write(data)

    # Calculate the hash of the downloaded data
    sha256_hash = hashlib.sha256(data).hexdigest()

    # Compare the calculated hash to the expected hash
    if sha256_hash != expected_hash:
        print('@@@ Downloaded file hash does not match expected hash!')
        raise RuntimeError

    return file_path


def build_data(resource, directory='data'):
    # check whether the file already exists
    if resource.filename.endswith('.tar.gz'):
        resource_dir = os.path.splitext(os.path.splitext(os.path.basename(resource.filename))[0])[0]
    else:
        resource_dir = os.path.splitext(os.path.basename(resource.filename))[0]
    file_path = os.path.join(directory, resource_dir)

    built = check_built(file_path, resource.version)

    if not built:
        # Download the file
        file_path = download_and_check_hash(resource.url, resource.filename, resource.expected_hash, resource.version,
                                            directory)

        # Unzip the file
        if resource.zipped:
            built_location = unzip_file(file_path, directory)
            # Delete the zip file
            os.remove(file_path)
        else:
            built_location = file_path

        mark_built(built_location, resource.version)
        print("Successfully built dataset at {}".format(built_location))
    else:
        print("Already built at {}. version {}".format(file_path, resource.version))
        built_location = file_path

    return built_location


def load():
    dpath = build_data(FANTOM)
    file = os.path.join(dpath, "fantom_v1.json")
    df = pd.read_json(file)

    return df


def get_prompt_format(data: dict) -> str:
    document = data["document"]
    question = data["question"]
    prompt = f"Conversation: {document}\n\nQuestion: {question}\n" \
             "Answer the given question based on the conversation." \
             "When you answer, think carefully. At the end, you MUST write the answer as an integer inside the squared brackets."
    input_ids = tokenizer.apply_chat_template(
        [{
            "role": "user",
            "content": prompt,
        }],
        padding=False,
        add_generation_prompt=True,
    )
    return input_ids


if __name__ == "__main__":
    df = load()
    # use 90% of the data for training
    train_df = df.sample(frac=0.9)
    # use the rest for testing
    test_df = df.drop(train_df.index)



    # reformat data for easier use.
    # document, question, correct_answer, wrong_answer
    def reformulate(row) -> list[dict]:

        # document = row.full_context
        document = row.short_context
        belief_qas = row.beliefQAs
        data = []
        for qa in belief_qas:
            question = qa["question"]
            correct_answer = qa["correct_answer"]
            wrong_answer = qa["wrong_answer"]
            data.append({"document": document,
                         "question": question,
                         "correct_answer": correct_answer,
                         "wrong_answer": wrong_answer})
        return data


    train_data = [reformulate(row) for _, row in train_df.iterrows()]
    train_data = [item for sublist in train_data for item in sublist]
    test_data = [reformulate(row) for _, row in test_df.iterrows()]
    test_data = [item for sublist in test_data for item in sublist]

    # read json files to get train_data and test_data

    print(f"No. of training samples: {len(train_data)}")
    print(f"No. of testing samples: {len(test_data)}")

    # example
    print(train_data[0])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    filtered_train_data = []

    tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-128k-instruct') 
    # find the statistics of the train_data, by tokenizing train_data[i]['document'] and train_data[i]['question']. Find max length of both, and max of summation. Also find the max length of the answers. give every statistics (mean, q3)
    length_document = []
    length_question = []
    length_answer = []
    length_sum = []
    for train in train_data:
        doc_len = len(tokenizer(train['document'])['input_ids'])
        question_len = len(tokenizer(train['question'])['input_ids'])
        answer_len = max(len(tokenizer(train['correct_answer'])['input_ids']), len(tokenizer(train['wrong_answer'])['input_ids']))
        total_len = doc_len + question_len
        
        # Only keep samples where the sum of document and question length is less than or equal to 1500
        if total_len <= 900:
            length_document.append(doc_len)
            length_question.append(question_len)
            length_answer.append(answer_len)
            length_sum.append(total_len)
            filtered_train_data.append(train)

    # if length_sum is larger than 1500, delete that sample

    print(f"Statistics of the training data:")
    print(f"Mean length of document: {sum(length_document)/len(length_document)}")
    print(f"Mean length of question: {sum(length_question)/len(length_question)}")
    print(f"Mean length of answer: {sum(length_answer)/len(length_answer)}")
    print(f"Mean length of sum: {sum(length_sum)/len(length_sum)}")
    print(f"Q3 length of document: {sorted(length_document)[int(len(length_document)*0.75)]}")
    print(f"Q3 length of question: {sorted(length_question)[int(len(length_question)*0.75)]}")
    print(f"Q3 length of answer: {sorted(length_answer)[int(len(length_answer)*0.75)]}")
    print(f"Q3 length of sum: {sorted(length_sum)[int(len(length_sum)*0.75)]}")
    print(f"Max length of document: {max(length_document)}")
    print(f"Max length of question: {max(length_question)}")
    print(f"Max length of answer: {max(length_answer)}")
    print(f"Max length of sum: {max(length_sum)}")
    ### train data length sum plot (barplot)
    plt.hist(length_sum, bins=30)
    plt.title("Length of document + question")
    plt.savefig("length_sum_train_data.png")
    # example
            
    # do the same stuffs for test_data
    filtered_test_data = []
    length_document = []
    length_question = []
    length_answer = []
    length_sum = []
    for test in test_data:
        doc_len = len(tokenizer(test['document'])['input_ids'])
        question_len = len(tokenizer(test['question'])['input_ids'])
        answer_len = max(len(tokenizer(test['correct_answer'])['input_ids']), len(tokenizer(test['wrong_answer'])['input_ids']))
        total_len = doc_len + question_len
        
        # Only keep samples where the sum of document and question length is less than or equal to 1500
        if total_len <= 1200:
            length_document.append(doc_len)
            length_question.append(question_len)
            length_answer.append(answer_len)
            length_sum.append(total_len)
            filtered_test_data.append(test)
            
    print(f"Statistics of the testing data:")
    print(f"Mean length of document: {sum(length_document)/len(length_document)}")
    print(f"Mean length of question: {sum(length_question)/len(length_question)}")
    print(f"Mean length of answer: {sum(length_answer)/len(length_answer)}")
    print(f"Mean length of sum: {sum(length_sum)/len(length_sum)}")
    print(f"Q3 length of document: {sorted(length_document)[int(len(length_document)*0.75)]}")
    print(f"Q3 length of question: {sorted(length_question)[int(len(length_question)*0.75)]}")
    print(f"Q3 length of answer: {sorted(length_answer)[int(len(length_answer)*0.75)]}")
    print(f"Q3 length of sum: {sorted(length_sum)[int(len(length_sum)*0.75)]}")
    print(f"Max length of document: {max(length_document)}")
    print(f"Max length of question: {max(length_question)}")
    print(f"Max length of answer: {max(length_answer)}")
    print(f"Max length of sum: {max(length_sum)}")
    ### test data length sum plot (barplot)
    plt.hist(length_sum, bins=30)
    plt.title("Length of document + question")
    plt.savefig("length_sum_test_data.png")
    print(len(filtered_train_data))
    print(len(filtered_test_data))
    

    import json 
    # save train-data and test-data
    with open("train_data_FANTOM.json", "w") as f:
        json.dump(filtered_train_data, f, indent=4)
    with open("test_data_FANTOM.json", "w") as f:
        json.dump(filtered_test_data, f, indent=4)
        