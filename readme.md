# Readme for the assignments.

The assignments are divided into 2 parts. One looks at annotation and agreement (corresponding notebook: annotation.ipynb, corresponding file:
assignment1.txt) and one at evaluation (corresponding notebook: evaluation.ipynb, corresponding file:
assignment2.txt). For each
assignment, there is a Jupyter notebook that contains some code snippets and questions. THe same questions are also in the corresponding .txt file.
Please answer the questions in the .txt file and upload them to Ilias into the corresponding assignment folder. You can also upload a pdf file if you prefer.

**The assignments are due to 9th of November.** 

The files are also shared via Github: https://github.com/Blubberli/course_disagreements

### Do I have to pogramm? 
There is little to program in these assignments. Most of the code can be run as is and most questions are about your conceptual understanding.
However, you may want to change some data or numbers to see how the metrics behave. In the last question you need to test some metrics on the given data, so 
a little bit of programming is required there.

### Do I have to write a lot?
No, bullet points or short answers are fine. You don't have to write long essays.

### Do I have to do everything correctly to pass?
No, the assignments are meant to help you understand the material. They are a simple pass/fail. If you make a mistake, you will get feedback.

# How to run the code?

This guide will help you set up the necessary environment to run the Jupyter notebooks or the Python scripts that were created from them. Follow the steps below carefully.

## Step 1: Installing Python

Before anything else, make sure you have Python installed. I am using Python 3.9, it is recommended to use the same version (or a compatible one).

## Step 2: Setting Up a Virtual Environment

A virtual environment helps keep your project dependencies (the packages and libraries you need) separate from your main system. Here’s how to set one up.

1. **Open a terminal/command prompt**:
   - On macOS or Linux, open the terminal.

2. **Navigate to your project folder**:
   Use the `cd` command to go to the folder where your project files are. Example:
   ```bash
   cd /path/to/your/project
   ```

3. **Create a virtual environment**:
   Run the following command to create a virtual environment:
   ```bash
   python3.9 -m venv venv
   ```
   This will create a folder named `venv` in your project directory.

4. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

5. **Install the project requirements**:
   After activating the virtual environment, install the required packages by running:
   ```bash
   pip install -r requirements.txt
   ```

## Step 3: Running the Jupyter Notebook

Once the environment is set up, you can run the Jupyter notebooks as follows:

1. **Install Jupyter** (if it's not already installed):
   ```bash
   pip install jupyter
   ```

2. **Launch the Jupyter Notebook**:
   Run this command to start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. **Open the notebook**:
   After you run the command, a browser window will open showing the Jupyter interface. Navigate to the folder where your `.ipynb` file is located and click on it to open the notebook. If you have an environment such as PyCharm, you can also open the notebook there.

4. **Run the notebook**:
   Once the notebook is open, you can run the code cells by clicking on each cell and then hitting `Shift + Enter` to execute it.

## Step 4: Running Python Scripts Instead (Optional)

If you prefer not to use Jupyter notebooks or encounter any issues with it, you can also run the Python scripts that were generated from the notebooks.

1. **Locate the Python scripts**:
   The Python scripts are called the same but with a  `.py` ending and they that were created from the Jupyter notebooks.


## Troubleshooting

- **Command not found**: If you get an error saying "command not found" for any step, double-check your installation and ensure that you have Python and the required packages installed.
- **Kernel Issues in Jupyter**: If the notebook isn’t running or the kernel isn’t connecting, restart the kernel or try running the code in a fresh environment.

Feel free to ask questions or get help if you face any issues!