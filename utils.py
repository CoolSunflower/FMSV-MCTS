import os
import fitz
import pytesseract
from PIL import Image
import numpy as np
from config import *

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

def read_document(pdf_path):
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

def run_jg(weak_answer:str, SPEC_NAME, signal_name, i, toret='string'):
    import paramiko
    import re
    from dotenv import load_dotenv
    load_dotenv()

    # SSH server details
    hostname =  os.environ.get(f"HOST_NAME")
    port = 22
    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")

    # print(hostname, username, password)

    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    local_log_path = "temp.txt"

    ROOT = "/storage/ckarfa/hpdmc/"
    if SPEC_NAME == "hpdmc":
        remote_sv_file_path = ROOT + "sva/hpdmc/hpdmc.sv"
        remote_log_path = ROOT + "output.txt"
        sv_template = HPDMC_sv_template
        JG_COMMAND = """
            csh -c '
            source cad_cshrc
            cd hpdmc
            jg -batch hpdmc.tcl > output.txt
            '
        """
    elif SPEC_NAME == "i2c":
        remote_sv_file_path = ROOT + "sva/i2c/i2c.sv"
        remote_log_path = ROOT + "output.txt"
        sv_template = I2C_sv_template
        JG_COMMAND = """
            csh -c '
            source cad_cshrc
            cd hpdmc
            jg -batch i2c.tcl > output.txt
            '
        """
    elif SPEC_NAME == 'rv':
        remote_sv_file_path = ROOT + 'sva/rv/rv.sv'
        remote_log_path = ROOT + 'output.txt'
        sv_template = RV_sv_template
        JG_COMMAND = '''
            csh -c '
            source cad_cshrc
            cd hpdmc
            jg -batch rv.tcl > output.txt
            '
        '''

    print(remote_sv_file_path)

    def extract_svas(sv_code: str, parameter_string: str):
        # single_line_pattern = re.compile(r'assert property \(@\(.*?\) (.*?)\);', re.DOTALL)
        # Extract only valid single-line SVAs (ignoring empty property references)
        # single_line_svas = [f"assert property (@(posedge sys_clk) {sva});" for sva in single_line_pattern.findall(sv_code)]

        # Updated regex to strictly capture assertions with `@(...)`
        single_line_pattern = re.compile(r'assert property\s*\(@\([^)]*\)\s*.*?\);')
        # Extract only valid single-line SVAs while keeping their clock cycle
        single_line_svas = single_line_pattern.findall(sv_code)

        multi_line_pattern = re.compile(
            r'(property\s+\w+;.*?endproperty\s+assert property \(\w+\);)', 
            re.DOTALL
        )        
        # Extract full multi-line SVAs as a whole block
        multi_line_svas = multi_line_pattern.findall(sv_code)

        # Return only fully valid SVAs by appending with parameters
        ret = []
        for item in (single_line_svas + multi_line_svas):
            ret.append(parameter_string + item)

        return ret

    def extract_assertions(weak_answer):
        # Extract the SystemVerilog block
        code_block_matches = re.findall(r"```systemverilog(.*?)```", weak_answer, re.DOTALL)
        if not code_block_matches:
            return []
        
        code_block = code_block_matches[-1].strip()

        # Extract parameters (lines that start with "parameter")
        parameter_pattern = r"parameter\s+.*?;"
        parameters = re.findall(parameter_pattern, code_block, re.MULTILINE)
        parameter_string1 = ""
        for parameter in parameters:
            parameter_string1 += parameter + "\n"

        parameter_pattern = r"localparam\s+.*?;"
        parameters = re.findall(parameter_pattern, code_block, re.MULTILINE)
        parameter_string2 = ""
        for parameter in parameters:
            parameter_string2 += parameter + "\n"

        return extract_svas(code_block, parameter_string1 + parameter_string2)

    assertions = extract_assertions(weak_answer)

    for item in assertions:
        print(item)
        print("-" * 80)

    def modify_template(assertion_entry):
        """Insert parameters and assertion into the SystemVerilog template."""
        return sv_template.format(properties=f"{assertion_entry}")

    def execute_ssh_commands(ssh):
        """Runs the required commands in a single SSH session."""
        command = JG_COMMAND
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout.channel.recv_exit_status()  # Wait for execution

    try:
        ssh.connect(hostname, port, username, password)
        sftp = ssh.open_sftp()
        results = {}
        from time import time
        start = time()

        for assertion_entry in assertions:
            modified_sv = modify_template(assertion_entry)

            print("Getting Feedback from JG - Starting File Upload")

            # Upload the new SystemVerilog file
            with sftp.file(remote_sv_file_path, "w") as f:
                f.write(modified_sv)

            print("Getting Feedback from JG - File Upload Successful")

            execute_ssh_commands(ssh)

            print("Getting Feedback from JG - Command Execution Successful. Downloading Logs")

            sftp.get(remote_log_path, local_log_path)

            with open(local_log_path, "r") as log_file:
                log_content = log_file.read()

            # Store the log output mapped to the assertion
            results[assertion_entry] = log_content

        # Close connections
        sftp.close()
        ssh.close()
        end = time()

        print("Total Time Taken to get JG Feedback: ", end - start)

        import pickle 

        with open(f'JasperGold_Logs/{SPEC_NAME}/{signal_name}_{i}.pkl', 'wb') as f:
            pickle.dump(results, f)
            
    except Exception as e:
        print(f"Error: {e}")

    # Now I only need to pass on the assertions that gave syntax error here and also check if I need to continue
    toContinue = False

    # This values are just for logging and maybe removed if needed
    CompleteCorrect = 0
    SyntaxCorrectSemanticsWrong = 0
    SyntaxWrong = 0

    for key, value in results.items():
        if "=\nSUMMARY\n=" in value:
            # Look for the "- cex                       : " pattern
            match = re.search(r"- cex\s+:\s+(\d)", value)
            
            if match:
                cex_value = int(match.group(1))  # Extract number (0 or 1)
                if cex_value == 0:
                    CompleteCorrect += 1
                else:
                    # semantically not correct according to the current RTL
                    toContinue = True
                    SyntaxCorrectSemanticsWrong += 1

            results[key] = "No Syntax Errors Found. Ensure semantics are correct wrt Specification"
        else:
            toContinue = True # Syntactically wrong assertions
            SyntaxWrong += 1

    print(f"Results Summary:\n\tSyntax Wrong: {SyntaxWrong}\n\tSyntax Correct Semantics Wrong: {SyntaxCorrectSemanticsWrong}\n\tComplete Correct: {CompleteCorrect}")

    if toret == 'string':
        return str(results), toContinue
    elif toret == 'dict':
        return results, toContinue
    
def getValidAssertions(spec_name, signal_name, document_data, llm_link, api_key):
    """
    It takes as input the spec name, signal name and document_data and does the following:
        1) Opens all the outputs from LLMOutput Folder which have generated assertions 
                (it opens the following files: LLMOutput/{spec_name}_{signal_name}_{i for i in range(5)_GetBetterAnswer.txt} and also 
                                               LLMOutput/{spec_name}_{signal_name}_S_WeakAnswer.txt)
        2) Extracts all the assertions from all the files into a long string (This is done in the following steps:
                            i) For each file find the last occurence of the text `LLM Output:`, first extract the content after this
                            ii) Then from this extracted string, find and extract whatever is in between the last ```systemverilog and ``` tags.)
        3) Sorts them into two sets using jaspergold, either syntax correct or incorrect
        4) Passes the syntax incorrect to an LLM along with document_data for one round of corrections
        5) Combines all assertions and passes them along with document_data to an LLM for extraction of all the unique ones 
        6) The resulting assertions are printed on the screen and returned as a string
    """
    import os
    import re
    from openai import OpenAI

    def extract_assertions_from_file(file_path):
        """Extracts SystemVerilog assertions from a given file."""
        with open(file_path, "r", encoding = 'utf-8') as f:
            content = f.read()
        
        # Find the last occurrence of "LLM Output:" and extract content after it
        llm_output_match = content.rsplit("LLM Output:", 1)
        if len(llm_output_match) < 2:
            return []  # No valid output found
        extracted_content = llm_output_match[-1].strip()
        
        # Extract assertions from within ```systemverilog and ``` tags
        code_block_matches = re.findall(r"```systemverilog(.*?)```", extracted_content, re.DOTALL)
        return [code_block_matches[-1].strip()] if code_block_matches else []

    def classify_assertions_with_jaspergold(assertions, spec_name, signal_name):
        """Runs assertions through JasperGold and classifies them as correct or incorrect."""
        results, _ = run_jg("```systemverilog\n" + "\n".join(assertions) + "\n```", spec_name, signal_name, 'final', toret = 'dict')
        
        syntax_correct = []
        syntax_incorrect = []
        reason_incorrect = []
        
        for assertion, result in results.items():
            if "No Syntax Errors Found" in result:
                syntax_correct.append(assertion)
            else:
                syntax_incorrect.append(assertion)
                reason_incorrect.append(result)
        
        return syntax_correct, syntax_incorrect, reason_incorrect

    def correct_syntax_errors(incorrect_assertions, reason_incorrect, document_data):
        """Passes incorrect assertions to LLM for correction."""
        prompt = f"Using the following documentation for reference return the assertions (provided later) with their syntax issue fixed: {document_data}. Here are the SystemVerilog assertions with syntax errors:\n"
        for i in range(len(incorrect_assertions)):
            prompt += incorrect_assertions[i] + '\n'
            prompt += reason_incorrect[i] + '\n'
        prompt += f"\nOnly give your output between the ```systemverilog and ``` tags."
        prompt += f"Please correct the syntax of the assertions provided earlier and return only the corrected assertions for that signal: {signal_name}. Ensure that you do not use the else construct. You are allowed to have parameters if needed. The only constrainst being you are not allowed to create any new signals than the ones specified in the specification."
        prompt += f"Note that you can add parameter declarations using localparam if missing."

        client = OpenAI(
            base_url=llm_link,
            api_key=api_key,
        )

        full_response = ""

        # Stream the response
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=128000,
            stream=True  # Enable streaming
        )

        with open(f"LLMOutput/{spec_name}_{signal_name}_final_SyntaxCorrection.txt", "w") as text_file:  # Open in append mode
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token  # Store token in memory
                    text_file.write(token)  # Write token to file
                    text_file.flush()  # Ensure immediate writing

        # Extract assertions from within ```systemverilog and ``` tags
        code_block_matches = re.findall(r"```systemverilog(.*?)```", full_response, re.DOTALL)
        return [match.strip() for match in code_block_matches] if code_block_matches else []

    def extract_svas(sv_code: str, parameter_string: str):
        # single_line_pattern = re.compile(r'assert property \(@\(.*?\) (.*?)\);', re.DOTALL)
        # Extract only valid single-line SVAs (ignoring empty property references)
        # single_line_svas = [f"assert property (@(posedge sys_clk) {sva});" for sva in single_line_pattern.findall(sv_code)]

        # Updated regex to strictly capture assertions with `@(...)`
        single_line_pattern = re.compile(r'assert property\s*\(@\([^)]*\)\s*.*?\);')
        # Extract only valid single-line SVAs while keeping their clock cycle
        single_line_svas = single_line_pattern.findall(sv_code)

        multi_line_pattern = re.compile(
            r'(property\s+\w+;.*?endproperty\s+assert property \(\w+\);)', 
            re.DOTALL
        )        
        # Extract full multi-line SVAs as a whole block
        multi_line_svas = multi_line_pattern.findall(sv_code)

        # Return only fully valid SVAs by appending with parameters
        ret = []
        for item in (single_line_svas + multi_line_svas):
            ret.append(parameter_string + item)

        return ret

    def extract_assertions(weak_answer):
        # Extract the SystemVerilog block
        code_block_matches = re.findall(r"```systemverilog(.*?)```", weak_answer, re.DOTALL)
        if not code_block_matches:
            return []
        
        code_block = code_block_matches[-1].strip()

        # Extract parameters (lines that start with "parameter")
        parameter_pattern = r"parameter\s+.*?;"
        parameters = re.findall(parameter_pattern, code_block, re.MULTILINE)
        parameter_string1 = ""
        for parameter in parameters:
            parameter_string1 += parameter + "\n"

        parameter_pattern = r"localparam\s+.*?;"
        parameters = re.findall(parameter_pattern, code_block, re.MULTILINE)
        parameter_string2 = ""
        for parameter in parameters:
            parameter_string2 += parameter + "\n"

        return extract_svas(code_block, parameter_string1 + parameter_string2)

    # def deduplicate_assertions(assertions, document_data):
    #     """Uses LLM to remove duplicate assertions and ensure consistency."""
    #     prompt = f"""
    #     Using the following documentation for reference:
    #     {document_data}
        
    #     Here are several SystemVerilog assertions:
    #     {assertions}
        
    #     Extract all unique and valid assertions from this list for the signal name: {signal_name}. Ensure that you maximise the number of retained assertions!! This maximisation is of utmost importance.
    #     Do not attempt to modify the actual assertion in any way, just return me all the unique assertions between the ```systemverilog and ``` tags.
    #     """

    #     client = OpenAI(
    #         base_url=llm_link,
    #         api_key=api_key,
    #     )

    #     full_response = ""

    #     # Stream the response
    #     stream = client.chat.completions.create(
    #         model=MODEL_NAME,
    #         messages = [
    #             {"role": "system", "content": "You are a helpful assistant"},
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=0.0,
    #         max_tokens=128000,
    #         stream=True  # Enable streaming
    #     )

    #     with open(f"LLMOutput/{spec_name}_{signal_name}_final_DeDuplication.txt", "w") as text_file: 
    #         for chunk in stream:
    #             if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
    #                 token = chunk.choices[0].delta.content
    #                 full_response += token  # Store token in memory
    #                 text_file.write(token)  # Write token to file
    #                 text_file.flush()  # Ensure immediate writing

    #     return full_response

    def deduplicate_assertions(assertions, document_data):
        """Uses LLM to remove duplicate assertions and ensure consistency."""
        prompt = f"""
        Using the following documentation for reference:
        {document_data}
        
        Here are several SystemVerilog assertions:
        {assertions}
        
        Extract all unique and valid assertions from this list for the signal name: {signal_name}. Ensure that you maximise the number of retained assertions!! This maximisation is of utmost importance.
        Very very important instructions to follow: Do not attempt to modify the actual assertion in any way, just return me all the unique assertions between the ```systemverilog and ``` tags.
        Don't forget any assertion (width, connectivity, functionality). Ensure that the assertions for width of current signal is always present, add that assertion if missing.
        """

        client = OpenAI(
            api_key=api_key,
            base_url=llm_link
        )

        full_response = ""

        # Stream the response
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            stream=True  # Enable streaming
        )

        with open(f"LLMOutput/{spec_name}_{signal_name}_final_DeDuplication.txt", "w") as text_file:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token  # Store token in memory
                    text_file.write(token)  # Write token to file
                    text_file.flush()  # Ensure immediate writing

        return full_response

    def process_assertions(spec_name, signal_name, document_data):
        """Main function to extract, validate, correct, and deduplicate assertions."""
        folder_path = "LLMOutput"
        file_patterns = [
            f"{spec_name}_{signal_name}_{i}_GetBetterAnswer.txt" for i in range(5)
        ] + [f"{spec_name}_{signal_name}_S_WeakAnswer.txt"]
        
        assertions = []
        count = 0

        # Extract assertions from files
        for file_name in file_patterns:
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                assertions.extend(extract_assertions_from_file(file_path))
                count += 1

        print(f"Sucessfully parsed {count} files")
        if not assertions:
            return "No assertions found."
        
        # Classify assertions using JasperGold
        syntax_correct, syntax_incorrect, reason_incorrect = classify_assertions_with_jaspergold(assertions, spec_name, signal_name)
        
        # Correct syntax errors if any exist
        if syntax_incorrect:
            corrected_assertions = correct_syntax_errors(syntax_incorrect, reason_incorrect, document_data)
            corrected_assertions = extract_assertions("```systemverilog\n" + corrected_assertions[-1] + "\n```")
        else:
            corrected_assertions = []
        
        # # Combine all assertions and remove duplicates
        with open(f'FinalOutputs/{spec_name}/{signal_name}_Duplicates.txt', 'w') as file:
            for item in (syntax_correct + corrected_assertions):
                file.write(item)
                file.write('\n\n')

        final_assertions = deduplicate_assertions(syntax_correct + corrected_assertions, document_data)

        with open(f'FinalOutputs/{spec_name}/{signal_name}_NoDuplicatesLLM.txt', 'w') as file:
            file.write(final_assertions)

        return final_assertions
    
    return process_assertions(spec_name, signal_name, document_data)