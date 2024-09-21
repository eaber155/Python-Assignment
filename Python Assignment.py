# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 21:38:29 2022

@author: eaber
"""

# import pandas and numpy libraris for dealing with tabular datasets
import pandas as pd
from numpy import genfromtxt

# import bokeh libraries for visualization
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row
from bokeh.models import LinearAxis, Range1d

# import sqlalchemy libraries for database manipulation
import sqlalchemy
import sqlalchemy as db

# import math library for mathematical computations
import math

# import csv reader for reading csv files line by line
from csv import reader

# import sys, traceback and time libraries for reading standard exceptions
import sys
import traceback
import time

# import unittest library for performing unit tests
import unittest


def standard_exception_details():
    '''
    
    Provides details for all standard exceptions
    
    Parameters
    -------
    None

    Returns
    -------
    exception_info : string
        A long string which contains the following details:
            1. Program line number where exception has occurred
            2. The procedure which threw the exception
            3. The exception/ error type
            4. The exception message showing details of the exception
            5. The program code which threw the exception
            6. The time stamp when the exception happened
            7. The file name and path where the exception occurred
    
    '''
    
    try:
        # obtain the exception details and assign them to the exception_info variable with additional information
        exception_type, exception_value, exception_traceback = sys.exc_info()
        file_name, line_number, procedure_name, line_code = traceback.extract_tb(exception_traceback)[-1]
        exception_info = ''.join('Standard Exception(Error Details): \n' + 'Program line number: ' 
        + str(line_number) + ' '+'\n'
        + 'Procedure Name: ' + str(procedure_name) + ' '+'\n' 
        + 'Error Type: ' + str(exception_type) + ' '+'\n'
        + 'Error Message: ' + str(exception_value) + ' '+'\n'  
        + 'Program code: ' + str(line_code) + '' + '\n'
        +'Time Stamp: ' + str(time.strftime('%d-%m-%Y %I:%M:%S %p'))+ '' +'\n' 
        + 'File Name: ' + str(file_name) + '' + '\n')
    except:
        pass
    return exception_info


class DefinedException(Exception):
    '''
    Provides details of a user defined exception
    
    Exception: The parent Exception class which DefinedException inherits from
    
    '''
    def __init__(self, exception_parameter, exception_message):
        super().__init__(self, exception_parameter, exception_message)
  

def import_csv(file_path):
    '''
    
    Imports any csv file and loads to a python dataframe
    
    Parameters
    ----------
    file_path : string
        Holds the file path for the CSV file to be uploaded

    Returns
    -------
    file : pandas dataframe
        Pandas dataFrame with the data which has been uploaded from the CSV file

    '''
    
    try:
        file = pd.read_csv(file_path)
        return file
    except:
        exception_details = standard_exception_details()
        print(exception_details)
    finally:
        pass
    

class IterateDataset():
    '''
    Builds an iterator from a column of any given dataframe
    '''
    
    def __init__(self, df, start, data_size, func_number):
        '''
        Initializing the class
        
        Parameters
        ----------
        df : pandas dataframe
            Pandas dataframe to be iterated
        start : integer
            Starting point of the iterator. Where the iterator should start
        data_size : integer
            Number of rows in the pandas dataFrame
        func_number : integer
            The function number indicating the column in the dataframe to be iterated

        Returns
        -------
        None.

        '''
        self.df = df
        self.start  = start-1
        self.data_size = data_size-1
        self.func_number = func_number
        
    def __iter__(self):
        '''
        Iterator
        
        Returns
        -------
        Iterator
        
        '''
        return self
    
    def __next__(self):
        '''
        Returns next item in the iterator
        
        Raises
        ------
        StopIteration
            Raises StopIteration exception when you reach the end of the iterator

        Returns
        -------
        Next item in the iterator
            Iterator with values of the next item in the column

        '''
        
        if self.start < self.data_size:
            self.start += 1
            return self.df.iloc[:,self.func_number][self.start]
        else:
            raise StopIteration
                   

class ComputeDeviation():
    '''
    
    Compute the deviation between the ideal function and the training function
    
    '''
    
    def __init__(self, train_df, ideal_df, no_of_train_funcs, no_of_ideal_funcs, start, data_size):
        '''
        
        Initializing the ComputeDeviation class
        
        Parameters
        ----------
        train_df : pandas dataFrame
            The train functions dataFrame
        ideal_df : pandas dataFrame
            The ideal functions dataFrame
        no_of_train_funcs : integer
            The number of train functions in train_df
        no_of_ideal_funcs : integer
            The number of ideal functions in ideal_df
        start : integer
            The starting point of the data/ iterator
        data_size : Integer
            The number of rows in each of the dataFrames

        Returns
        -------
        None.

        '''

        self.train_df = train_df
        self.ideal_df = ideal_df
        self.no_of_train_funcs = no_of_train_funcs
        self.no_of_ideal_funcs = no_of_ideal_funcs
        self.start = start
        self.data_size = data_size
           
    def derive_ideal_function (self):
        '''
        
        Derives the ideal function based on the deviation criteria.
        [The criterion for choosing the ideal functions for the training function is how they 
        minimize the sum of all y-deviations squared (Least-Square)]

        Returns
        -------
        chosen_functions : Dictionary
            Dictionary containing the train function and it's derived ideal function

        '''
        
        try:
            chosen_functions = {}
            
            # perform procedure for each train function
            for t in range(0, self.no_of_train_funcs):
                deviations = []
                # perform procedure for each ideal function
                for i in range(0,self.no_of_ideal_funcs):
                    total_deviation = 0
                    # iterate over the train function and the ideal function
                    train_function = IterateDataset(self.train_df,self.start,self.data_size,t+1)
                    ideal_function = IterateDataset(self.ideal_df,self.start,self.data_size,i+1)
                    
                    # compute the deviation between each ideal data point and train data point using sum of square of the difference
                    for ideal_, train_ in zip(ideal_function,train_function):
                        deviation = (ideal_-train_)**2
                        total_deviation += deviation
                    # append the deviation for each of the ideal functions to the deviations list
                    deviations.append(total_deviation)
                
                # get the ideal function with the least deviation and obtain its function number as the index of the deviations list
                ideal_func_index=deviations.index(min(deviations))+1
                
                # add to the chosen_functions dictionary with the train function as the key and the chosen ideal function index as the value
                chosen_functions[t+1] = ideal_func_index
                
            return chosen_functions
        except:
            # return a standard exception incase of any errors
            exception_details = standard_exception_details()
            print(exception_details)
        finally:
            pass  



def plot_func(title, x,  x_label, df1=None, df2=None, y1=None, y2=None, y_label1=None, y_label2=None, y_label3=None,
             title_align='center', sizing_mode='scale_width', legend_location='top_center'):
    '''
    
    Creates a bokeh graph using the train functions dataframe or ideal functions Dataframe or both
    Able to plot the train functions, ideal functions and a combination both of them

    Parameters
    ----------
    title : string
        This is the bokeh HTML file path/name
    x : string
        Name of the x columns in the dataframe
    x_label : string
        Label for the x-axis on the graph
    df1 : pandas dataframe, optional
        The dataframe with the train functions. 
        The default is None.
    df2 : pandas dataframe, optional
        The dataframe with the ideal functions. 
        The default is None.
    y1 : string, optional
        Name of the column containing the function being plotted from the train functions dataFrame. 
        The default is None.
    y2 : string, optional
        Name of the column containing the function being plotted from the ideal functions dataFrame. 
        The default is None.
    y_label1 : string, optional
        Label for the y-axis on the graph plotting the train functions. 
        The default is None.
    y_label2 : string, optional
        Label for the y-axis on the graph plotting the ideal functions. 
        The default is None.
    y_label3 : string, optional
        Label for the y-axis on the graph plotting a combination of the train function and the ideal function. 
        The default is None.
    title_align : string, optional
        Alignment of the title against the graph. 
        The default is 'center'.
    sizing_mode : string, optional
        Size scaling in relation to the page size. 
        The default is 'scale_width'.
    legend_location : string, optional
        Location of the legend in relation to the graph. 
        The default is 'top_center'.

    Returns
    -------
    func : bokeh figure
        Graph which has been plotted

    '''
    
    try:
        if(y1!=None and y2==None):
            # plot the train or ideal functions datasets
            func = figure(title=title, x_axis_label=x_label, y_axis_label=y_label1)
            func.line(df1[x], df1[y1], legend_label=y_label1, line_width=2)
        else:
            # plot a combination of the train or ideal functions datasets
            func = figure(title=title, x_axis_label = x_label, y_axis_label=y_label3)
            func.line(df1[x], df1[y1], legend_label=y_label1, line_width=2)
            func.line(df2[x], df2[y2], legend_label=y_label2, line_width=2, line_color='orange')
            
        # align the title, define the sizing mode and the legend location
        func.title.align= title_align
        func.sizing_mode = sizing_mode
        func.legend.location = legend_location
        return func
    except:
        # return a standard exception incase any error occurs
        exception_details = standard_exception_details()
        print(exception_details)
    finally:
        pass        


def plot_ideal_n_train_func(file_name, title1, title2, title3, df1, df2, x, y1, y2, x_label, y_label1, y_label2, y_label3,
                           title_align='center', sizing_mode='scale_width', legend_location='top_center'):
    '''
    
    - Adds three bokeh graphs to an HTML page: 1 Graph with the train function, the second graph with the corresponding ideal function 
    and the third graph which has a combination of the train and ideal functions.
    - Takes the same parameters as the plot_func() since it uses the plot_func() to create the bokeh graphs and then renders them to a
    page in an HTML file.

    Parameters
    ----------
    title : string
        This is the bokeh HTML file path/name
    x : string
        Name of the x columns in the dataframe
    x_label : string
        Label for the x-axis on the graph
    df1 : pandas dataFrame, optional
        The dataframe with the train functions. 
        The default is None.
    df2 : pandas dataFrame, optional
        The dataframe with the ideal functions.
        The default is None.
    y1 : string, optional
        Name of the column containing the function being plotted from the train functions dataframe. 
        The default is None.
    y2 : string, optional
        Name of the column containing the function being plotted from the ideal functions dataframe. 
        The default is None.
    y_label1 : string, optional
        Label for the y-axis on the graph plotting the train functions. 
        The default is None.
    y_label2 : string, optional
        Label for the y-axis on the graph plotting the ideal functions. 
        The default is None.
    y_label3 : string, optional
        Label for the y-axis on the graph plotting a combination of the train function and the ideal function. 
        The default is None.
    title_align : string, optional
        Alignment of the title against the graph. 
        The default is 'center'.
    sizing_mode : string, optional
        Size scaling in relation to the page size. 
        The default is 'scale_width'.
    legend_location : string, optional
        Location of the legend in relation to the graph. 
        The default is 'top_center'.

    Returns
    -------
    None.

    '''
    try:
        # define the output file name
        output_file(file_name)
        # plot1 - plots the train data using the plot_func function
        train = plot_func(title1, x, x_label, df1=df1, y1=y1, y_label1=y_label1, title_align=title_align, 
                          sizing_mode=sizing_mode, legend_location=legend_location)
        # plot2 - plots the ideal data using the plot_func function
        ideal = plot_func(title2, x, x_label, df1=df2, y1=y2, y_label1=y_label2, title_align=title_align, 
                          sizing_mode=sizing_mode, legend_location=legend_location)
        # plot3 - plots the train data and ideal data on the same axis using the plot_func function
        train_n_ideal = plot_func(title3, x, x_label, df1=df1, df2=df2, y1=y1, y2=y2, y_label1=y_label1, 
                                  y_label2=y_label2, y_label3=y_label3,title_align=title_align, 
                          sizing_mode=sizing_mode, legend_location=legend_location)
        
        # renders plot1, plot2 and plot3 on the same page 
        plot = row(train, ideal, train_n_ideal)
        plot.sizing_mode='scale_width'
        show(plot)
    except:
        # returns a standard error incase of any exceptions
        exception_details = standard_exception_details()
        print(exception_details)
    finally:
        pass  


def Load_Data(file_name):
    '''
    Loads a file from CSV and converts it to a list

    Parameters
    ----------
    file_name : string
        String containing the file path and name

    Returns
    -------
    dataset_list : list
        List containing the values which were loaded from the excel file

    '''
    
    dataset = genfromtxt(file_name, delimiter=',', skip_header=1)
    dataset_list = dataset.tolist()
    return dataset_list


def db_connection(db_name, password, account):
    '''
    Function for connecting to a MySQL database

    Parameters
    ----------
    db_name : string
        The name of the database
    password : string
        Password credentials for the database management system
    account : string
        Account name for the database management system

    Raises
    ------
    DefinedException
        Raises an exception if the database does not connect, indicating that the credentials used are incorrect.

    Returns
    -------
    engine : sqlalchemy engine
        Sqlalchemy engine for creating a connection to the database
    connection : sqlalchemy connection
        Sqlalchemy connection for working with the database and running SQL functions

    '''
    
    try:
        engine = db.create_engine("mysql+pymysql://"+ account + ":"+ password + "@localhost/"+ db_name)
        connection = engine.connect()
        return engine, connection
    except:
        # if an error occurs, raises a defined exception indicating that authentication details are incorrect
        raise DefinedException(db_name, 'Incorrect authentication details for {} database.')


def Drop_Table(table, connection):
    
    '''

    Parameters
    ----------
    table : sqlalchemy table
        Table to be dropped from the database
    connection : sqlalchemy Connection
        Sqlalchemy connection returned by db_connection for working with the database and running SQL statements

    Returns
    -------
    None.

    '''
    
    try:
        sql = "DROP TABLE eunice_32111626_db."+table
        connection.execute(sql)
    except:
        # print warning if error is experienced
        print('{} table does not exist or is not visible to a connection.'.format(table))
    finally:
        pass


def Create_Train_Table(engine, connection):
    '''
    
    Creates the train table in the database for storing train functions data

    Parameters
    ----------
    engine : sqlalchemy engine
        Sqlalchemy engine returned by db_connection function for creating a connection to the database
    connection : Sqlalchemy connection
        Sqlalchemy connection returned by db_connection function for working with the database and running SQL statements

    Returns
    -------
    train_dataset : sqlalchemy table
        The train database table for storing the train functions data

    '''
    
    meta_data = db.MetaData()
    Drop_Table('train_dataset', connection)
    train_dataset= db.Table(
        "train_dataset", meta_data,
        db.Column('id', db.Integer, primary_key=True, autoincrement=True, nullable=False),
        db.Column('X', db.Float,nullable=False),
        db.Column('Y1', db.Float, nullable=False), db.Column('Y2', db.Float, nullable=False),
        db.Column('Y3', db.Float, nullable=False), db.Column('Y4', db.Float, nullable=False)
    )
    
    meta_data.create_all(engine)
    print('Train Table created')
    return train_dataset


def Create_Ideal_Table(engine, connection):
    '''
    
    Creates the ideal table in the database for storing ideal functions data

    Parameters
    ----------
    engine : sqlalchemy engine
        Sqlalchemy engine returned by db_connection function for creating a connection to the database
    connection : sqlalchemy connection
        Sqlalchemy connection returned by db_connection function for working with the database and running SQL statements

    Returns
    -------
    ideal_dataset : sqlalchemy table
        The ideal database table for storing the ideal functions data

    '''
    
    meta_data = db.MetaData()
    Drop_Table('ideal_dataset', connection)
    ideal_dataset = db.Table(
        "ideal_dataset", meta_data,
        db.Column('id', db.Integer, primary_key=True, autoincrement=True, nullable=False),
        db.Column('X', db.Float,nullable=False),
        db.Column('Y1', db.Float, nullable=False), db.Column('Y2', db.Float, nullable=False),
        db.Column('Y3', db.Float, nullable=False), db.Column('Y4', db.Float, nullable=False),
        db.Column('Y5', db.Float, nullable=False), db.Column('Y6', db.Float, nullable=False),
        db.Column('Y7', db.Float, nullable=False), db.Column('Y8', db.Float, nullable=False), 
        db.Column('Y9', db.Float, nullable=False), db.Column('Y10', db.Float, nullable=False), 
        db.Column('Y11', db.Float, nullable=False), db.Column('Y12', db.Float, nullable=False), 
        db.Column('Y13', db.Float, nullable=False), db.Column('Y14', db.Float, nullable=False), 
        db.Column('Y15', db.Float, nullable=False), db.Column('Y16', db.Float, nullable=False), 
        db.Column('Y17', db.Float, nullable=False), db.Column('Y18', db.Float, nullable=False), 
        db.Column('Y19', db.Float, nullable=False), db.Column('Y20', db.Float, nullable=False), 
        db.Column('Y21', db.Float, nullable=False), db.Column('Y22', db.Float, nullable=False), 
        db.Column('Y23', db.Float, nullable=False), db.Column('Y24', db.Float, nullable=False),
        db.Column('Y25', db.Float, nullable=False), db.Column('Y26', db.Float, nullable=False),
        db.Column('Y27', db.Float, nullable=False), db.Column('Y28', db.Float, nullable=False), 
        db.Column('Y29', db.Float, nullable=False), db.Column('Y30', db.Float, nullable=False), 
        db.Column('Y31', db.Float, nullable=False), db.Column('Y32', db.Float, nullable=False), 
        db.Column('Y33', db.Float, nullable=False), db.Column('Y34', db.Float, nullable=False), 
        db.Column('Y35', db.Float, nullable=False), db.Column('Y36', db.Float, nullable=False), 
        db.Column('Y37', db.Float, nullable=False), db.Column('Y38', db.Float, nullable=False), 
        db.Column('Y39', db.Float, nullable=False), db.Column('Y40', db.Float, nullable=False), 
        db.Column('Y41', db.Float, nullable=False), db.Column('Y42', db.Float, nullable=False), 
        db.Column('Y43', db.Float, nullable=False), db.Column('Y44', db.Float, nullable=False), 
        db.Column('Y45', db.Float, nullable=False), db.Column('Y46', db.Float, nullable=False), 
        db.Column('Y47', db.Float, nullable=False), db.Column('Y48', db.Float, nullable=False), 
        db.Column('Y49', db.Float, nullable=False), db.Column('Y50', db.Float, nullable=False)
    )

    meta_data.create_all(engine)
    print('Ideal Table created')
    return ideal_dataset



def insert_train_or_ideal_values(connection, file_name, table, max_range):
    '''
    
    Inserts the train functions data or the ideal functions data into their respective database tables

    Parameters
    ----------
    connection : sqlalchemy connection
        Sqlalchemy Connection returned by db_connection function for working with the database and running SQL statements.
    file_name : string
        String containing the file path and name
    table : string
        Name of the database table where data is being inserted
    max_range : string
        Maximum number of column which are contained in the data

    Returns
    -------
    result : Sqlalchemy Engine Cursor
        Sqlchemy cursor returned when a connection is made

    '''
    
    file_name = file_name
    # upload the data from csv and convert to list using the Load_Data function
    data_list = Load_Data(file_name)
    
    sql_query1= db.insert(table)

    dataset_list = []

    # add the data to a dictionary with the keys are the column names and the values are the data point
    for i in data_list:
        dataset_dict = {}
        dataset_dict['X'] = i[0]
        for y in range(1,max_range):
            key = "Y"+str(y)
            dataset_dict[key] = i[y]
        # append each dictionary to the dataset_list 
        dataset_list.append(dataset_dict)
        
    # upload the dataset list to the database using the connection.execute function
    result = connection.execute(sql_query1, dataset_list)
    return result


class MaxDeviation(ComputeDeviation):
    '''
    Compute the maximum deviation between the ideal function and its correspondin train function
    '''
    
    def __init__(self, train_df, ideal_df, no_of_train_funcs, no_of_ideal_funcs, start, data_size):
        '''
        
        Initialse the class

        Parameters
        ----------
        train_df : pandas dataFrame
            The dataFrame containing the train functions
        ideal_df : pandas dataFrame
            The dataFrame containing the ideal functions
        no_of_train_funcs : integer
            The number of functions in the train functions dataset
        no_of_ideal_funcs : integer
            The number of functions in the ideal functions dataset
        start : integer
            The starting row for the iterator object
        data_size : integer
            The number of rows in the dataset

        Returns
        -------
        None.

        '''
        
        # initialize MaxDeviation class
        super().__init__(train_df, ideal_df, no_of_train_funcs, no_of_ideal_funcs, start, data_size)
        

    def compute_max_deviation (self):
        '''
        
        Compute the maximum deviation between the ideal function and its corresponding train function
        
        Parameter
        -------
        None 
        
        Returns
        -------
        maximum_deviations : dictionary
            Dictionary containing each ideal function and its maximum deviation with the corresponding 
            train function

        '''
        
        try:
            chosen_functions = self.derive_ideal_function()
            maximum_deviations = {}
                
            # perform this procedure for each of the chosen ideal functions
            for key in chosen_functions:
                all_deviations = []
                
                # iterate the train function and the ideal function
                train_function = IterateDataset(self.train_df,self.start,self.data_size,key)
                ideal_function = IterateDataset(self.ideal_df,self.start,self.data_size,chosen_functions[key])
                    
                '''
                    - for each data point in the train and ideal function, compute the deviation using the square of the difference 
                between the y values
                    - store each deviation in the all_deviations list
                '''
                for ideal_, train_ in zip(ideal_function,train_function):
                    deviation = (ideal_-train_)**2
                    all_deviations.append(deviation)
                
                # obtain the ideal function number
                ideal_func = 'y'+str(chosen_functions[key])
                # obtain the maximum deviation from the deviations list for each ideal function and store in a dictionary
                maximum_deviations[ideal_func] = max(all_deviations)
                
            # return a dictionary containing all ideal functions with the ideal function as the key and its maximum deviation as the value
            return maximum_deviations
        except:
            # return a standard exception in case of any errors
            exception_details = standard_exception_details()
            print(exception_details)
        finally:
            pass  
    
    

def Create_Test_Table(engine, connection):
    '''
    
    - Creates the mapped test table for storing the test data which is mapped to the corresponding ideal function(s) that meet the criteria,
    and stores the corresponding deviation
    - Deviation is computed as the square of the difference between the ideal function and the corresponding train function

    Parameters
    ----------
    engine : sqlalchemy engine
        Sqlalchemy engine returned by db_connection function for creating a connection to the Database
    connection : sqlalchemy connection
        Sqlalchemy connection returned by db_connection function for working with the database and running SQL statements

    Returns
    -------
    mapped_test_data : sqlalchemy table
        The mapped test database table for storing the test data, its mapped ideal function(s) that meet the criteria, 
        and their corresponding deviation

    '''
    
    meta_data = db.MetaData()
    Drop_Table('mapped_test_data', connection)
    mapped_test_data = db.Table(
        "mapped_test_data", meta_data,
        db.Column('id', db.Integer, primary_key=True, autoincrement=True, nullable=False),
        db.Column('X', db.Float,nullable=False),
        db.Column('Y', db.Float, nullable=False), db.Column('Delta Y', db.String(100), nullable=False),
        db.Column('No. of ideal func', db.String(100), nullable=False)
    )

    meta_data.create_all(engine)
    print('Test Table created')
    return mapped_test_data


def MapFunctions(engine, connection, maximum_deviations, mapped_test_data,  test_data_file, ideal_functions_df):
    '''
    Loads the test data from a CSV file line by line. 
    - For each row of test data, computes the deviation between the test function and each ideal function
    - For each row of test data, chooses the ideal function(s) with the lowest deviation
    - Deviation is computed as the square of the difference between the test function and the ideal function
    - Uploads the mapped test points to the database

    Parameters
    ----------
    engine : sqlalchemy engine
        Sqlalchemy engine returned by db_connection function for creating a connection to the Database
    connection : sqlalchemy connection
        Sqlalchemy connection returned by db_connection function for working with the database and running SQL statements
    maximum_deviations : dictionary
        Dictionary containing the maximum deviations for each ideal function computed using the MaxDeviation.compute_max_deviation function
    mapped_test_data : sqlalchemy table
        Sqlalchemy table for storing the mapped test data
    test_data_file : string
        Holds the file path for the CSV file containing the test data
    ideal_functions_df : pandas dataframe
        Pandas dtaframe holdining ideal functions dataset

    Returns
    -------
    result : Sqlalchemy Engine Cursor
        Sqlchemy cursor returned when a connection is made
    mapped_functions : Array
        An array which contains a dictionary that has each test datapoint, its corresponding ideal function, 
        and the deviation between the test function and the ideal function

    '''
    
    try:
        # initiliaze mapped_functions list to store mapped test points
        mapped_functions = []
    
        # open the csv file which contains the test points dataset
        with open(test_data_file,'r') as test_file:
            # read the test data
            test_dataset = reader(test_file)
            # get the next test data point header
            header = next(test_dataset)
    
            # perform testing if there is a data point (i.e. the header is not null)
            if header != None:
                for test_point in test_dataset:
                    # extract the four ideal functions which correspond to the test points and add then extract the y values into a dictionary
                    relevant_ideal_functions= ideal_functions_df[ideal_functions_df['x']==float(test_point[0])]
                    ideal_functions_dict = relevant_ideal_functions.loc[:,relevant_ideal_functions.columns != 
                                                                        'x'].to_dict('records')[0]
                    # get the test point
                    test_point_value = float(test_point[1])
                    # initialize the deviation dictionary
                    deviation_dict = {} 
                    
                    '''
                     - for each of the ideal functions, compute the deviation between the test point and the ideal functions
                     - the deviation is computed as the square of the difference between the test point y value and the ideal function y value
                    '''      
                    for ideal_func, ideal_func_value in ideal_functions_dict.items():
                        deviation = (test_point_value-ideal_func_value)**2
                        deviation_dict[ideal_func] = deviation
                        
                        # initiliaze the success and chosen_deviations dictionary to store mapping status for the ideal functions and their deviations
                        success = {}
                        chosen_deviations = {}
                        
                        '''
                          add each ideal function to the success dictionary 
                          - indicate "yes" where the ideal function has been successfully mapped to the test  point
                          - indicate "no" where the ideal function has not been successfully mapped to the test point
                          - an ideal function is mapped sucessfully to a test point if the deviation computed is less than 
                          or equal to the maximum deviation of the ideal function from the train function
                          
                        '''
                        # add each of the ideal functions 
                        for func, deviation in deviation_dict.items():
                            if deviation <= (maximum_deviations[func]*math.sqrt(2)):
                                # add the ideal function as key and the mapping status as "yes" if it has been mapped successfully
                                success[func] = 'yes'
                                # add an additional chosen_deviations dictionary with the deviation for the ideal functions which have been successfull mapped
                                chosen_deviations[func] = deviation
                            else:
                                # add the ideal function as key and the mapping status as "no" if it has not been mapped successfully
                                success[func] = 'no'
    
                    # add the successfully mapped ideal functions into a list
                    mapped_ideal_functions = [key for key,value in success.items() if value == 'yes']
                    # add the deviations for the successfully mapped ideal functions into a list
                    corresponding_deviations = [value for key,value in chosen_deviations.items()]
    
                    '''
                        - add the test X value, Y value and the corresponding mapped ideal function and the computed deviation to the mapped_test_data 
                        database table if the mapped_ideal_functions list is not empty

                        - add the test X value, Y value and "Not applicable", "No applicable function" to the mapped_test_data database table if
                        if the mapped_ideal_functions list is empty
                        
                    '''
                    
                    if len(mapped_ideal_functions) > 0:
                        for i in range(0, len(mapped_ideal_functions)):
                            mapped_data = {}
                            mapped_data['X'] = float(test_point[0])
                            mapped_data['Y'] = float(test_point[1])
                            mapped_data['Delta Y'] = corresponding_deviations[i]
                            mapped_data['No. of ideal func'] = mapped_ideal_functions[i]
                            insert_row_values = []
                            insert_row_values.append(mapped_data)
                            sql_query1= db.insert(mapped_test_data)
                            result = connection.execute(sql_query1, insert_row_values)
                            mapped_functions.append(mapped_data)
                    else:
                        mapped_data  ={}
                        mapped_data['X'] = float(test_point[0])
                        mapped_data['Y'] = float(test_point[1])
                        mapped_data['Delta Y'] = "Not applicable"
                        mapped_data['No. of ideal func'] = "No applicable function"
                        insert_row_values = []
                        insert_row_values.append(mapped_data)
                        sql_query1= db.insert(mapped_test_data)
                        result = connection.execute(sql_query1, insert_row_values)
                        mapped_functions.append(mapped_data)
                        
        # return the database cursor and the mapped_test_dataset
        return result, mapped_functions
    except:
        # return a standard exception incase of any errors
        exception_details = standard_exception_details()
        print(exception_details)
    finally:
        pass  


def extract_test_points(df, function_number):
    '''
    Extracts all tests points based on the ideal function it is mapped to.
    - Assigns a prefix 'yellow_' for all points which were mapped to that function number
    - Assigns a prefix 'red_' for all points which were not mapped to that function

    Parameters
    ----------
    df : DataFrame
        A Dataframe containing all the test points and their mapped ideal functions and  corresponding deviations
    function_number : String
        ID for the ideal function e.g. 'y19'

    Returns
    -------
    dataset_list_func : Array
        An array with all test points mapped to that function
    dataset_list_not_func : Array
        An array with all test points not mapped to that function

    '''
    
    def create_test_points_dict(function_df, test_points_color):
        '''
        Creates a dictionary with the test points and their corresponding colors. 

        Parameters
        ----------
        function_df : pandas dataframe
            A dataframe containing the test functions to be extracted (either those mapped to the function or those not mapped to the function)
        test_points_color : string
            A string containing the color which the test point will be plotted as

        Returns
        -------
        dataset_dict : dictionary
            A dictionary containing the test points extract. 
            The color and position of the test point is the key, and the list containing the x and y values of the test point as the dictionary value

        '''
        
        # add extracted test points to a dictionary
        dataset_dict = {}
        # initialize count to store the position of the test point in the dictionary
        count = 1
        for i in function_df.itertuples():
            # initialize the list for storing the x and y values fo the test point
            test_points = []
            # obtain the x data point in the mapped test point
            test_points.append(i[1])
            # obtain the y data point in the mapped test point
            test_points.append(i[2])
            
            # if the test point is not mapped to the function, add 'red' as a suffix to its position number
            color = test_points_color + "_" + str(count)
            # add each test point to a dictionary with its color and position as the key and the test point list as the value
            dataset_dict[color] = test_points
            count+=1 
        return dataset_dict
        

    # extract test_points which are mapped to the function stored in the function_number argument
    test_points_df_func = df[df['No. of ideal func']==function_number]
    
    # extract test_points which are not mapped to the function stored in the function_number argument
    test_points_df_not_func = df[df['No. of ideal func']!=function_number]

    # add extracted test points to corresponding dictionaries
    # if the test point is mapped to the function, add 'yellow' as a suffix to its position number
    dataset_dict_func = create_test_points_dict(test_points_df_func, "yellow")
    # if the test point is not mapped to the function, add 'red' as a suffix to its position number
    dataset_dict_not_func = create_test_points_dict(test_points_df_not_func, "red")
    
    # return the test point dictionaries containing test points which are mapped to the function and those not mapped
    return dataset_dict_func, dataset_dict_not_func



def plot_test_points(file_name, title, df, x, x_label, y, y_label, 
                     y_range=[-2200,500], title_align='center', 
                     plot_width=1000, plot_height=600, dot_size=10, **kwargs):
    '''
    Plots the ideal function and all the test points extracted using the extract_test_points function
    - Uses bokeh library and renders the graphs to an HTML file

    Parameters
    ----------
    file_name : string
        File name/path for the HTML file where the bokeh graph will be rendered
    title : string
        The title of the graph being plotted
    df : pandas dataframe
        A dataframe containing the ideal function
    x : string
        The x variable name in the ideal function
    x_label : string
        The label for the x-axis in the ideal function graph
    y : string
        The y variable name in the ideal function (the function being plotted)
    y_label : string
        The label for the y-axis in the ideal function graph
    y_range : list, optional
        The scale range for the secondary y-axis 
        The default is [-2200,500].
    title_align : string, optional
        The alignment of the title in relation to the graph. 
        The default is 'center'.
    plot_width : integer, optional
        The setting for the width of the graph. 
        The default is 1000.
    plot_height : integer, optional
        The setting for the height of the graph. 
        The default is 600.
    dot_size : integer, optional
        The setting for the dot size which represents a test point on the graph. 
        The default is 10.
    **kwargs : dictionary
        Kwargs arguments to take in the test points which use the scale for the primary axis and those which use the 
        scale for the secondary axis

    Returns
    -------
    None.

    '''
    
    try:
        # define the output HTML file using its name
        output_file(file_name)
        # define the bokeh figure for plotting the ideal function and the test points
        plot_func = figure(title=title, x_axis_label=x_label, y_axis_label=y_label)
        # plot ideal function in the primary axis
        primary_axis = plot_func.line(df[x], df[y], line_width=2)
        
        # plot all test points with the color 'yellow' in their key on the primary axis i.e all test points mapped to that ideal function)
        all_primary = []
        all_primary.append(primary_axis)
        if len(kwargs['primary']) != 0:
            for key, value in kwargs['primary'].items():
                primary_test_point = plot_func.circle_dot([value[0]], [value[1]], fill_color=key.split('_')[0], size=dot_size)
                all_primary.append(primary_test_point)
        # render the test points plot to the bokeh figure
        plot_func.y_range.renderers=all_primary
        
        # add a secondary axis to the bokeh figure
        plot_func.extra_y_ranges = {"secondary":Range1d(start=y_range[0], end=y_range[1])}
        plot_func.add_layout(LinearAxis(y_range_name='secondary'), 'right')
        
        # plot all test points with the color 'red' in their key on the secondary axis i.e all test points not mapped to that ideal function
        if len(kwargs['secondary']) != 0:
            for key, value in kwargs['secondary'].items():
                plot_func.circle_dot([value[0]], [value[1]], fill_color=key.split('_')[0], size=dot_size, y_range_name='secondary')
            
        # align title
        plot_func.title.align=title_align
        # define figure width      
        plot_func.plot_width = plot_width
        # define figure height
        plot_func.plot_height = plot_height
        
        show(plot_func)
    except:
        # return standard exception if any errors occur in this process
        exception_details = standard_exception_details()
        print(exception_details)
    finally:
        pass
    
    
class TestImportantElements(unittest.TestCase):
    '''
    
    Unit tests designed for important parts of the code
    
    '''

    # upload train dataset to a pandas dataframe    
    train_dataset_df = import_csv(r'Datasets\train.csv')
    # upload ideal datase to a pandas dataframe
    ideal_dataset_df = import_csv(r'Datasets\ideal.csv')
    
    # define database parameters
    db_name = 'eunice_32111626_db'
    password = '*namusokeZ25*'
    account = 'eaber155'
    
    # initiliaze the engine and connection for the database using the db_connection function
    engine, connection = db_connection(db_name, password, account)

    def test_correctidealfunctions(self):
        '''
        
        Unit test for checking that the correct ideal functions have been selected based on the criteria

        Returns
        -------
        None.
        

        '''
        
        # compute the deviation between ideal functions and the train functions using the ComputeDeviation.compute_deviation function
        compute_deviation = ComputeDeviation(self.train_dataset_df, self.ideal_dataset_df, 4, 50, 0, 400)
        derived_ideal_functions = compute_deviation.derive_ideal_function()
        
        # define the expected results from the ComputeDeviation.compute_deviation function
        expected_result = {1: 19, 2: 46, 3: 28, 4: 43}
        
        # test the assertion for the compute_deviation function computation process by checking if the chosen ideal functions are as expected
        for train, ideal in derived_ideal_functions.items():
            self.assertEqual(ideal, expected_result[train], "Ideal functions not correct. Check train dataset")
        
        
    def test_maximum_deviation(self):
        '''
        
        Unit test for checking that the maximum deviations have been computed correctly based on the 
        deviation criteria

        Returns
        -------
        None.

        '''
        
        # compute maximum deviation between test point and the ideal functions
        max_deviation = MaxDeviation(self.train_dataset_df, self.ideal_dataset_df, 4, 50, 0, 400)
        maximum_deviations = max_deviation.compute_max_deviation()
        
        # define expected results for maximum deviation
        expected_result = {'y19': 0.24828295839996575, 'y46': 0.24783032801536053, 
                              'y28': 0.24966011560000576,'y43': 0.2490784508217601}
        
        # test the assertion for the maximum_deviations functions computation process by checking if the maximum deviations are as expected
        for function, result in maximum_deviations.items():
            self.assertEqual(result, expected_result[function], 
                             "Maximum deviation for function {} is incorrect".format(function))

    
    def test_mapped_functions(self):
        '''
        
        Unit test for checking if the test data points have been correctly mapped

        Raises
        ------
        DefinedException
            Raises an input exception if the ComputeDeviation.compute_deviation function doesn't execute
            correctly

        Returns
        -------
        None.

        '''
        
        # derive the ideal functions based on the least-square criterion
        try:
            compute_deviation = ComputeDeviation(self.train_dataset_df, self.ideal_dataset_df, 4, 50, 0, 400)
            ideal_functions_dict = compute_deviation.derive_ideal_function()
        except:
            raise DefinedException(None, 'Check accuracy of inputs supplied')
            
        
        # add the ideal functions names to a list
        ideal_functions_list = []
        ideal_functions_list.append('x')
        for key, value in ideal_functions_dict.items():
            ideal_functions_list.append('y'+str(value))
        
        # extract ideal functions from the ideal_dataset dataframe
        try:
            ideal_functions = self.ideal_dataset_df[ideal_functions_list].copy()
        except:
            exception_details = standard_exception_details()
            print(exception_details)
        finally:
            pass
        
        # compute maximum deviation between the test point and the ideal functions
        max_deviation = MaxDeviation(self.train_dataset_df, self.ideal_dataset_df, 4, 50, 0, 400)
        maximum_deviations = max_deviation.compute_max_deviation()
        
        # create the mapped_test_data table in the database using the Create_Test_Table function
        mapped_test_data = Create_Test_Table(self.engine, self.connection)
        
        # map each test point to the ideal functions using the MapFunctions function
        result, mapped_functions = MapFunctions(self.engine, self.connection, maximum_deviations, mapped_test_data, 
                                                r'Datasets\test.csv', ideal_functions)
        
        # create a dataframe with the mapped test points dataframe
        mapped_functions_df = pd.DataFrame(mapped_functions)
        
        # obtain the shape of the mapped test points dataframe where the test points have not been mapped to any function
        unmapped_functions_shape = mapped_functions_df[mapped_functions_df['No. of ideal func']==
                                                       'No applicable function'].shape
        
        # test the assertion for the test point-ideal functions mapping process by checking if the shape of the unmapped test points dataframe is as expected
        self.assertEqual(unmapped_functions_shape, (68, 4), "Mapping not correct. Review mapping criteria")
        
    
    def test_traindb(self):
        '''
        
        Unit test to check if the data has been correctly loaded to the train_dataset database table

        Returns
        -------
        None.

        '''
        
        # initialize meta data
        meta_data = db.MetaData()
        # return the train_dataset table
        train_dataset = db.Table("train_dataset", meta_data, autoload=True, autoload_with=self.engine)
        # select everything from the train_dataset table in the database
        select_train_data = db.select([train_dataset])
        dataset = self.connection.execute(select_train_data).fetchall()
        # get the shape of the database table and add to a list (first element as the number of rows and second element as the number of columns)
        table_shape = [len(dataset), len(dataset[0])]
        # test the assertion for the train_dataset database table insert process by checking if the number of rows and columns in the train_dataset database table are as expected
        self.assertEqual(table_shape, [400,6], "Incorrect train functions data upload")
        
        
    def test_idealdb(self):
        '''
        Unit test to check if the data has been correctly loaded to the train_dataset database table

        Returns
        -------
        None.

        '''
        
        # initialize meta data
        meta_data = db.MetaData()
        # return the ideal_dataset table
        ideal_dataset = db.Table("ideal_dataset", meta_data, autoload=True, autoload_with=self.engine)
        # select everything from the ideal_dataset table in the database
        select_ideal_data = db.select([ideal_dataset])
        dataset = self.connection.execute(select_ideal_data).fetchall()
        # get the shape of the database table and add to a list (first element as the number of rows and second element as the number of columns)
        table_shape = [len(dataset), len(dataset[0])]
        # test the assertion for the ideal_dataset table insert process by checking if the number of rows and columns in the ideal_dataset database table are as expected
        self.assertEqual(table_shape, [400,52], "Incorrect ideal functions data upload")
        
        
    def test_testdb(self):
        '''
        Unit test to check if the data has been correctly loaded to the train_dataset database table

        Returns
        -------
        None.

        '''
        # initialize meta data
        meta_data = db.MetaData()
        # return the mapped_test_data table
        mapped_test_data = db.Table("mapped_test_data", meta_data, autoload=True, autoload_with=self.engine)
        # select everything from the mapped_test_data table in the database
        select_test_data = db.select([mapped_test_data])
        dataset = self.connection.execute(select_test_data).fetchall()
        # get the shape of the database table and add to a list (first element as the number of rows and second element as the number of columns)
        table_shape = [len(dataset), len(dataset[0])]
        # test the assertion for the mapped_test_dataset table insert process by checking if the number of rows and columns in the mapped_test_data database table are as expected
        self.assertEqual(table_shape, [101,5], "Incorrect mapped test data upload")
            
    

def main():
    
    print("Reading the train and ideal datasets from CSV to a Pandas Dataframe.....")
    print('')
    train_dataset_df = import_csv(r'Datasets\train.csv')
    ideal_dataset_df = import_csv(r'Datasets\ideal.csv')
    
    '''
    
    Determining the ideal functions
    The criterion for choosing the ideal functions for the training function is how they minimize the sum of all y-
    deviations squared (Least-Square)
    
    '''
    
    print("Choosing the ideal functions from the ideal dataset based on the train dataset.....")
    print('--------------------------------------------------------------------------------')
    # derive the ideal functions based on the least-square criterion
    try:
        compute_deviation = ComputeDeviation(train_dataset_df, ideal_dataset_df, 4, 50, 0, 400)
    except:
        raise DefinedException(None, 'Check accuracy of inputs supplied')
        
    ideal_functions_dict = compute_deviation.derive_ideal_function()
        
    
    '''
    
    Visualising Train and Ideal Functions
    
    '''
    
    print("Visualizing the chosen ideal functions.....")
    print('--------------------------------------------------------------------------------')
    # visualize the first ideal function with its corresponding train function
    try:
        plot_ideal_n_train_func('Function y19.html', 'Train Dataset', 'Function y19 Dataset', 'Train and Function y19 Dataset',
                               train_dataset_df, ideal_dataset_df, 'x', 'y1', 'y19', 'x', 'Train Function (y1)', 
                                'Ideal Function (y19)', 'Train and Function y19 Dataset')
    except:
        raise DefinedException(None, 'Check accuracy of inputs supplied')
    

    #visualize the second ideal function with its corresponding train function
    try:
        plot_ideal_n_train_func('Function y46.html', 'Train Dataset', 'Function y46 Dataset', 'Train and Function y46 Dataset',
                               train_dataset_df, ideal_dataset_df, 'x', 'y2', 'y46', 'x', 'Train Function (y2)', 
                                'Ideal Function (y46)', 'Train and Function y46 Dataset', legend_location='bottom_right')
    except:
        raise DefinedException(None, 'Check accuracy of inputs supplied')
    

    #visualize the third ideal function with its corresponding train function
    try:
        plot_ideal_n_train_func('Function y28.html', 'Train Dataset', 'Function y28 Dataset', 'Train and Function y28 Dataset',
                               train_dataset_df, ideal_dataset_df, 'x', 'y3', 'y28', 'x', 'Train Function (y3)', 
                                'Ideal Function (y28)', 'Train and Function y28 Dataset', legend_location='bottom_right')
    except:
        raise DefinedException(None, 'Check accuracy of inputs supplied')

    #visualize the fourth ideal function with its corresponding train function
    try:
        plot_ideal_n_train_func('Function y43.html', 'Train Dataset', 'Function y43 Dataset', 'Train and Function y43 Dataset',
                               train_dataset_df, ideal_dataset_df, 'x', 'y4', 'y43', 'x', 'Train Function (y4)', 
                                'Ideal Function (y43)', 'Train and Function y46 Dataset', legend_location='bottom_right')
    except:
        raise DefinedException(None, 'Check accuracy of inputs supplied')
    
    
    '''
    
    Saving the train and ideal datasets to their respective database tables
    
    '''
    
    print("Creating a connection to the database.....")
    print('')
    # define database parameters
    db_name = 'eunice_32111626_db'
    password = '*namusokeZ25*'
    account = 'eaber155'
    
    # create a connection to the database by initiating the database engine and connection using the db_connection function
    engine, connection = db_connection(db_name, password, account)
    
    # create the train functions table in the database
    print("Creating the train database table.....")
    print('')
    train_dataset = Create_Train_Table(engine, connection)

    # upload the train functions dataset to the database using the insert_train_or_ideal_values function
    print("Insert the train data to the train database table.....")
    print('--------------------------------------------------------------------------------')
    try:
        insert_training_data = insert_train_or_ideal_values(connection, r'Datasets\train.csv', train_dataset, 5)
        print(insert_training_data)
    except:
        # user-defined exception raised incase any of the inputs are inaccurate
        raise DefinedException(None, 'Check accuracy of inputs supplied')
    
    
    # create the ideal functions table in the database
    print("Creating the ideal database table.....")
    print('')
    ideal_dataset = Create_Ideal_Table(engine, connection)
    
    # upload ideal functions dataset to the database using the insert_train_or_ideal_values function
    print("Insert the ideal data to the ideal database table.....")
    print('--------------------------------------------------------------------------------')
    try:
        insert_ideal_data = insert_train_or_ideal_values(connection, r"Datasets\ideal.csv", ideal_dataset, 51)
        print(insert_ideal_data)
    except:
        # user-defined exception raised incase any of the inputs are inaccurate
        raise DefinedException(None, 'Check accuracy of inputs supplied')
        
    
    '''
    
    Mapping the ideal functions to the test data
    The criterion for mapping the individual test case to the four ideal functions is that the existing maximum deviation- 
    of the calculated regression does not exceed the largest deviation between training dataset (A) and the ideal-
     function (C) chosen for it by more than factor sqrt(2)

    ##### NOTE: For the purpose of this computation, deviation between any two points is defined as square of the difference-
    between the two points 
    1. Evaluating maximum deviation = (y(indeal)-y(train))^2 
    2. Evaluating corresponding ideal function = (y(indeal) - y(train))^2
    
    '''
    
    # obtain a list of the chosen ideal functions from the ideal_functions_dict dictionary created by the ComputeDeviation.derive_ideal_function function
    print("Extracting a dataframe of the chosen ideal functions.....")
    print('')
    ideal_functions_list = []
    ideal_functions_list.append('x')
    for key, value in ideal_functions_dict.items():
        ideal_functions_list.append('y'+str(value))
    
    # extract a dataframe with only the chosen ideal functions
    try:
        ideal_functions = ideal_dataset_df[ideal_functions_list].copy()
    except:
        # standard exception returned incase of any error
        exception_details = standard_exception_details()
        print(exception_details)
    finally:
        pass
    
    
    # compute maximum deviation for the ideal functions using MaxDeviation.max_deviation function
    print("Obtaining the maximum deviation between the ideal functions and their corresponding train functions.....")
    print('')
    try:
        max_deviation = MaxDeviation(train_dataset_df, ideal_dataset_df, 4, 50, 0, 400)
        maximum_deviations = max_deviation.compute_max_deviation()
    except:
        # user-defined exception raised incase any of the inputs are inaccurate
        raise DefinedException(None, 'Check accuracy of inputs supplied')
    

    
    print("Creating the mapped_test_data table in the database.....")
    print('')
    # create mapped test table for storing test points and corresponding ideal functions
    mapped_test_data = Create_Test_Table(engine, connection)
    
    '''
    
    Map the test points to the corresponding ideal functions and insert in database using the MapFunctions function
    
    '''
    
    print("Mapping test data points to corresponding ideal functions and uploading results to the database.....")
    print('--------------------------------------------------------------------------------')
    try:
        result, mapped_functions = MapFunctions(engine, connection, maximum_deviations, mapped_test_data, 
                                                r'Datasets\test.csv', ideal_functions)
    except:
        # standard exception returned incase of any error
        exception_details = standard_exception_details()
        print(exception_details)
    finally:
        pass


    '''
    
    Visualising Ideal Functions with test datapoints
    
    '''
    
    print("Plotting the ideal functions, their mapped test points and unmapped test points.....")
    print('--------------------------------------------------------------------------------')
    
    mapped_functions_df = pd.DataFrame(mapped_functions)
    
    # extract the test points which are mapped to ideal function y19 and those not mapped to ideal function y19
    func_y19, not_func_y19 = extract_test_points(mapped_functions_df, 'y19')
    
    # plot ideal Function y19 with the test points not mapped to ideal function y19 and test points mapped to ideal function y19
    try:
        plot_test_points('Function y19_Test.html', 'Function y19 Dataset', ideal_dataset_df, 'x', 'x', 'y19','Train Function (y19)',
                        primary=func_y19, secondary=not_func_y19)
    except:
        # user-defined exception raised incase any of the inputs are inaccurate
        raise DefinedException(None, 'Check accuracy of inputs supplied')
        
    
    # extract the test points which are mapped to ideal function y46
    func_y46, not_func_y46 = extract_test_points(mapped_functions_df, 'y46')
    
    # plot ideal Function y46 with the test points not mapped to ideal function y46 and test points mapped to ideal function y46
    try:
        plot_test_points('Function y46_Test.html', 'Function y46 Dataset', ideal_dataset_df, 'x', 'x', 'y46','Train Function (y46)',
                         primary=func_y46, secondary=not_func_y46)
    except:
        # user-defined exception raised incase any of the inputs are inaccurate
        raise DefinedException(None, 'Check accuracy of inputs supplied')
        
    
    # extract the test points which are mapped to ideal function y28
    func_y28, not_func_y28 = extract_test_points(mapped_functions_df, 'y28')
    
    # plot ideal Function y28 with the test points not mapped to ideal function y28 and test points mapped to ideal function y28
    try:
        plot_test_points('Function y28_Test.html', 'Function y28 Dataset', ideal_dataset_df, 'x', 'x', 'y28','Train Function (y28)',
                         primary=func_y28, secondary=not_func_y28)
    except:
        # user-defined exception raised incase any of the inputs are inaccurate
        raise DefinedException(None, 'Check accuracy of inputs supplied')
        

    # extract the test points which are mapped to ideal function y43
    func_y43, not_func_y43 = extract_test_points(mapped_functions_df, 'y43')
    
    # plot ideal Function y43 with the test points not mapped to ideal function y43 and test points mapped to ideal function y43
    try:
        plot_test_points('Function y43_Test.html', 'Function y43 Dataset', ideal_dataset_df, 'x', 'x', 'y43','Train Function (y43)',
                        primary=func_y43, secondary=not_func_y43)
    except:
        # user-defined exception raised incase any of the inputs are inaccurate
        raise DefinedException(None, 'Check accuracy of inputs supplied')
        
    print("All done.....")


if __name__ == '__main__':
    # run main function
    main()
    
    print('Running unit tests.....')
    # run the unit tests
    unittest.main()
    


