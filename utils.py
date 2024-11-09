import os
from langchain_community.document_loaders import PyPDFLoader

def list_csv_files(directory):
    # List all CSV files in the specified directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        
    # Display the CSV files to the user
    print("Select a CSV file by entering its number:")
    for i, file_name in enumerate(csv_files):
        print(f"{i + 1}. {file_name}")
    
    try:
        choice = int(input("Enter the number of the file you want to select: "))
        if 1 <= choice <= len(csv_files):
            selected_file = csv_files[choice - 1]
            return selected_file[:-4] + '.pdf', selected_file[:-4]
        else:
            print("Invalid number. Please choose a valid number.")
            exit()
    except ValueError:
        print("Please enter a valid number.")
        exit()

def read_document(pdf):
    pdf_path = "./datasets/spec/" + pdf
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    data = ""
    data += "Source:" + documents[0].metadata['source'] + "\n"

    for i in range(len(documents)):
        data += "Page Number:" + str(documents[i].metadata['page']) + "\n"
        data += "Content:" + documents[i].page_content + "\n"

    return data