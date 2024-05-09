import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO
from datasets import Dataset

def modify_dataframe_and_extract_data(df):
    data_list = []
    for _, row in df.iterrows():
        messages = []
        for i in range(1, 5):
            user_question = row[f'Question{i}']
            user_answer = row[f'Answer{i}']
            if user_question:
                message_content = [{'index': None, 'text': user_question, 'type': 'text'}]
                if i == 1:
                    message_content.append({'index': 0, 'text': None, 'type': 'image'})
                messages.append({'content': message_content, 'role': 'user'})
                if user_answer:
                    messages.append({'content': [{'index': None, 'text': user_answer, 'type': 'text'}], 'role': 'assistant'})
        image = Image.open(row['imagePath'])
        data_list.append({'messages': messages, 'images': [image]})
    return {'messages': [data['messages'] for data in data_list], 'images': [data['images'] for data in data_list]}


def download_and_resize_images(df, image_dir, target_size=(250, 250)):
    image_paths = []
    for index, row in df.iterrows():
        image_url = row['primaryImageLink']
        object_id = row['objectID']
        if image_url:
            # Extract filename from the URL
            filename = os.path.join(image_dir, f"{object_id}.jpg")
            # Download image from the URL
            response = requests.get(image_url)
            if response.status_code == 200:
                # Open the image using PIL
                image = Image.open(BytesIO(response.content))
                # Resize the image
                image = image.resize(target_size)
                # Save the resized image
                image.save(filename)
                image_paths.append(filename)
            else:
                print(f"Failed to download image from {image_url}")
                image_paths.append(None)
        else:
            image_paths.append(None)
    return image_paths

def split_data_dict(data_dict, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    assert train_ratio + test_ratio + val_ratio == 1.0, "Ratios must sum up to 1.0"
    
    total_samples = len(data_dict['messages'])
    train_size = int(total_samples * train_ratio)
    test_size = int(total_samples * test_ratio)
    val_size = int(total_samples * val_ratio)

    train_data_dict = {
        'messages': data_dict['messages'][:train_size],
        'images': data_dict['images'][:train_size]
    }
    test_data_dict = {
        'messages': data_dict['messages'][train_size:train_size + test_size],
        'images': data_dict['images'][train_size:train_size + test_size]
    }
    val_data_dict = {
        'messages': data_dict['messages'][-val_size:],
        'images': data_dict['images'][-val_size:]
    }

    return train_data_dict, test_data_dict, val_data_dict


def save_data_dict_as_arrow(data_dict, file_path):
    # Convert the dictionary to a Dataset object
    dataset = Dataset.from_dict(data_dict)
    
    # Save the dataset to an Arrow file
    dataset.save_to_disk(file_path)

if __name__ == "__main__":
    # Example usage:

    # df = pd.read_csv("/data/data_set_metmuseum.csv")
    # df1 = df[['objectID', 'primaryImageLink', 'Question1', 'Answer1', 'Question2', 'Answer2', 'Question3', 'Answer3', 'Question4', 'Answer4']]
    # df2 = df1.sample(frac=1)
    # df3 = df2.head(250)

    # df4 = df3.copy()

    df4 = pd.read_csv("sampled_data250.csv")
    paths = ['input_dataset', os.path.join('input_dataset', 'images'), 'output_dataset']
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    # Call the function to download and resize images
    image_dir = 'input_dataset/images'
    # image_paths = download_and_resize_images(df4, image_dir)

    # Update the DataFrame with the resized image paths
    # new_df = df4.copy()  # Create a copy of the original DataFrame
    # df4['imagePath'] = image_paths  # Add a new column 'imagePath' containing the resized image paths
    # df4 = df4.drop(['primaryImageLink'], axis=1)

    # Call the function to modify the DataFrame and extract data
    data_dict = modify_dataframe_and_extract_data(df4)
    # split data_dict into train, test, valid
    train_data_dict, test_data_dict, val_data_dict = split_data_dict(data_dict, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2)

    # save these as arrow dataset
    save_data_dict_as_arrow(train_data_dict, os.path.join('output_dataset', 'train.arrow'))
    save_data_dict_as_arrow(test_data_dict, os.path.join('output_dataset', 'test.arrow'))
    save_data_dict_as_arrow(val_data_dict, os.path.join('output_dataset', 'val.arrow'))

    # save to zip format
    import shutil
    shutil.make_archive("/content/input_dataset", "zip", "/content/input_dataset")
    shutil.make_archive("/content/output_dataset", "zip", "/content/output_dataset")
    
    # read arrow from disk
    test_data = Dataset.load_from_disk("output_dataset/test.arrow")
    test_data
