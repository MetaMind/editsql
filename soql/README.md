# SOQL Annotations
This directory contains the work for creating Salesforce database schemas for input to the EditSQL model. The code for training the model is in the main directory and can be used to train a custom model for any SOQL data. It also contains the annotations from the S3 bucket that were collected from the worknbech chrome extension.

### Creating Schemas
Use the **create_tables.py** script to convert schema text files to EditSQL readable **tables.json** file.

Usage: `python create_tables.py database/ tables.json`

Here, the **database/** directory contains all the subdirectories which must contain the **schema.txt** file. These files are read by the script to finally output the **tables.json** file.
Sample schema [file](https://github.com/MetaMind/editsql/blob/master/soql/database/salesforce/schema.txt).

### Training Data
Use the script **parse_all_soql.py** to generate training and dev files based on random probability distribution for sampling the examples into the two sets.


Usage: `python parse_all_soql.py`

**parse_one_soql.py** shows an example on how to parse a SOQL query into its clause-level information. It uses the **parse_soql.py** script.
