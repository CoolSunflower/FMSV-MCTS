import os
import fitz
import pytesseract
from PIL import Image
import numpy as np

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

    # Open the PDF
    doc = fitz.open(pdf_path)
    zoom = 4  # Setting zoom level for better OCR accuracy
    mat = fitz.Matrix(zoom, zoom)
    
    # Directory for temporary image storage
    temp_folder = "./temp_images"
    os.makedirs(temp_folder, exist_ok=True)
    
    # Initialize the text accumulator
    full_text = ""

    # Convert each page to an image and perform OCR
    for page_num in range(doc.page_count):
        # Generate image filename
        img_path = os.path.join(temp_folder, f"page_{page_num + 1}.png")
        
        # Load page and convert to image
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)
        pix.save(img_path)

        # Open image and perform OCR
        img = Image.open(img_path)
        text = pytesseract.image_to_string(np.array(img))
        
        # Accumulate the extracted text
        full_text += text + "\n"
    
    # Close the PDF document
    doc.close()

    # Optional: Clean up temporary images
    for file_name in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, file_name)
        os.remove(file_path)
    os.rmdir(temp_folder)
    
    return full_text