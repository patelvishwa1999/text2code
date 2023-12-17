import pandas as pd
import re


def removeDocStrings(train_data):
  """
  removes Doc Strings from codes train_data having columns = [func_documentation_string,
                                                              func_code_string,
                                                              func_documentation_tokens,
                                                              func_code_tokens]

  """
  descriptions = []
  description_tokens = []
  codes = []
  code_tokens = []

  # Cleaning Up codes by removing doc strings
  for index, row in train_data.iterrows():
    curr_desc = row['func_documentation_string']
    curr_code = row['func_code_string']
    curr_description_token = row['func_documentation_tokens']
    curr_code_token = row['func_code_tokens']

    curr_clean_code = (curr_code.replace(curr_desc, '').replace("\"\"\"", ''))
    descriptions.append(curr_desc)
    codes.append(curr_clean_code)
    description_tokens.append(curr_description_token)
    code_tokens.append(curr_code_token)
  #creating pandas Dataframe to return
  df = pd.DataFrame({'func_documentation_string': descriptions, 'func_code_string': codes,
                     'func_documentation_tokens': description_tokens, 'func_code_tokens': code_tokens})

  return df



  def removeNonEnglish(train_data):

  """
  removes Non English codes from train_data having columns = [func_documentation_string,
                                                              func_code_string,
                                                              func_documentation_tokens,
                                                              func_code_tokens]

  """
    def contains_non_english(text):
        """
        helper function for removeNonEnglish()

        """
        # Defining a regular expression pattern to match non-English characters
        pattern = re.compile('[^\x00-\x7F]+')
        match = pattern.search(text)
        # If a match(True) is found, the string contains non-English characters
        return bool(match)


  descriptions = []
  description_tokens = []
  codes = []
  code_tokens = []

  for index, row in train_data.iterrows():
    curr_desc = row['func_documentation_string']
    curr_code = row['func_code_string']
    curr_description_token = row['func_documentation_tokens']
    curr_code_token = row['func_code_tokens']
    #if Non English strings found
    if contains_non_english(curr_desc) or contains_non_english(curr_code):
      pass
    #if English strings found
    elif not contains_non_english(curr_desc) and not contains_non_english(curr_code):
      descriptions.append(curr_desc)
      codes.append(curr_code)
      description_tokens.append(curr_description_token)
      code_tokens.append(curr_code_token)

  #creating pandas Dataframe to return
  df = pd.DataFrame({'func_documentation_string': descriptions, 'func_code_string': codes,
                     'func_documentation_tokens': description_tokens, 'func_code_tokens': code_tokens})

  return df


def add_start_end_tokens(df, columns):
    """
    adds <start> and <end> inplace
    """
    start_token = '<start>'
    end_token = '<end>'

    for column in columns:
        if column in df.columns:
            if 'tokens' in column:
                # For columns containing lists of tokens
                df[column] = df[column].apply(lambda x: x[0] + f"\'{start_token}\'" + " " + x[1:-1] + " " + f"\'{end_token}\'" + x[-1])
            else:
                # For columns containing strings
                df[column] = start_token + ' ' + df[column] + ' ' + end_token