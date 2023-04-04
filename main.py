
from fastapi import FastAPI, status, HTTPException, Depends, File, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from utils import get_hashed_password, verify_password, create_access_token, create_refresh_token
from deps import get_current_user
from database import Database
from schemas import UserAuth, UserOut, TokenSchema
import pandas as pd
from typing import List
from io import StringIO, BytesIO
from psycopg2 import errors
from fitparse import FitFile
from zipfile import ZipFile
import math
import os
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

app = FastAPI()

# Openai_API_key
api_key = os.environ.get('OPENAI_API_KEY')

# Create DB Schema if does not exist
with Database() as db:
    db.create_schema()
    db.close

# Function to parse FIT file
def parse_fit_file(fit_file, fields):
    fitfile = FitFile(fit_file)

    # Get data messages that are of type record
    records = fitfile.get_messages('record')

    # Create a list of dicts
    data = []
    # Go through all the records
    for record in records:
        # Initialize a dictionary with fields as keys and None as values
        record_data_dict = {field: None for field in fields}
        # Go through all the data entries in this record
        for record_data in record:
            # Check if the record's name is in the desired fields
            if record_data.name in fields:
                record_data_dict[record_data.name] = record_data.value
        # Append the record_data_dict to the data list
        data.append(record_data_dict)
    # Convert the list of dicts to a pandas DataFrame
    df = pd.DataFrame(data)
    return df

#------CREATE USER ENDPOINT----------------------------------------------------------------------

@app.post('/signup', summary="Create new user", tags=["Athlete"], response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(data: UserAuth):
    # querying database to check if user already exist
    with Database() as db:
        athlete = db.query_fetchone("SELECT username FROM public.athlete WHERE username = %s", (data.username,))
        db.close

    if athlete is not None:
        raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="User with this username already exist"

        )
    athlete = {
        'username': data.username,
        'password': get_hashed_password(data.password)
    }

    # save user to database
    with Database() as db:
        db.execute("INSERT INTO public.athlete (username, password) VALUES (%s, %s)", (athlete['username'], athlete['password']))
        db.close

    return athlete

#------DELETE USER ENDPOINT-----------------------------------------------------------------

@app.delete('/delete', summary="Delete user", tags=["Athlete"], status_code=status.HTTP_202_ACCEPTED)
async def delete_user(athlete: UserOut = Depends(get_current_user)):
    # delete user from database
    with Database() as db:
        db.execute("DELETE FROM public.athlete WHERE id = %s", (athlete['id'],))
        db.close

    return {"message": f"User account {athlete['username']} deleted successfully"}

#------UPDATE USER PASSWORD ENDPOINT-----------------------------------------------------------------

@app.put('/update', summary="Update password", tags=["Athlete"], status_code=status.HTTP_202_ACCEPTED)
async def update_user(password:str, athlete: UserOut = Depends(get_current_user)):
    with Database() as db:
        # update user password in database
        db.execute("UPDATE public.athlete SET password = %s WHERE id = %s", (get_hashed_password(password), athlete['id']))
        db.close

    return {"message": f"Password for user {athlete['username']} updated successfully"}

#------LOGIN ENDPOINT-----------------------------------------------------------------------

@app.post('/login', summary="Create access and refresh tokens for user", tags=["Athlete"], response_model=TokenSchema)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with Database() as db:
        athlete = db.query_fetchone("SELECT id,username,password FROM public.athlete WHERE username = %s", (form_data.username,))
        db.close
    
    if athlete is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )

    hashed_pass = athlete[2]
    if not verify_password(form_data.password, hashed_pass):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )
    
    return {
        "access_token": create_access_token(athlete[1]),
        "refresh_token": create_refresh_token(athlete[1]),
    }

#------RETURN CURRENT USER ENDPOINT-----------------------------------------------------------

@app.get('/me', summary='Get details of currently logged in user', tags=["Athlete"], response_model=UserOut)
#returning current user as dict
async def get_current_user(athlete: UserOut = Depends(get_current_user)):
    return athlete

#------GET FIT ACTIVITIES ENDPOINT------------------------------------------------------------------- 

# get list of activities from fit_data table. Group by activity_id and return averages for all columns except datetime for each activity.
@app.get('/fit_activities', summary="Get list of fit activities", tags=["Activities"])
def get_activities(athlete: UserOut = Depends(get_current_user)):
    with Database() as db:
        data = db.query_fetchall('''
        SELECT activity_id, AVG(speed), AVG(elevation), AVG(heartrate), AVG(cadence), AVG(power), 
               AVG(core_temperature), AVG(skin_temperature), AVG(stride_length), MIN(datetime), 
               MAX(datetime) 
        FROM public.fit_data 
        WHERE athlete_id = %s 
        GROUP BY activity_id 
        ORDER BY MIN(datetime) DESC
        ''', (athlete['id'],))
        db.close

    # return data as a list of dicts - ["activity_id":{1},"activity_id":{2},"activity_id":{3}"]
    data = [{"activity_id": row[0], 
             "start_time": row[9], 
             "end_time": row[10],
             "speed": row[1], 
             "elevation": row[2], 
             "heartrate": row[3], 
             "cadence": row[4], 
             "power": row[5], 
             "core_temperature": row[6], 
             "skin_temperature": row[7], 
             "stride_length": row[8] } for row in data]
    
    return data

#------GET CSV ACTIVITIES ENDPOINT----------------------------------------------------------------

# get list of activities from csv_data table. Group by activity_id and return averages for all columns except datetime for each activity.
@app.get('/csv_activities', summary="Get list of csv activities", tags=["Activities"])
def get_activities(athlete: UserOut = Depends(get_current_user)):
    with Database() as db:
        data = db.query_fetchall('''
        SELECT activity_id, AVG(speed), AVG(elevation), AVG(heartrate), AVG(cadence), 
               AVG(core_temperature), AVG(skin_temperature), AVG(stride_length), MIN(datetime), 
               MAX(datetime)    
        FROM public.csv_data 
        WHERE athlete_id = %s 
        GROUP BY activity_id 
        ORDER BY MIN(datetime) DESC
        ''', (athlete['id'],))
        db.close

    # return data as a list of dicts - ["activity_id":{1},"activity_id":{2},"activity_id":{3}"]
    data = [{"activity_id": row[0],
             "start_time": row[8],
             "end_time": row[9],
             "speed": row[1],
             "elevation": row[2],
             "heartrate": row[3],
             "cadence": row[4],
             "core_temperature": row[5],
             "skin_temperature": row[6],
             "stride_length": row[7] } for row in data]

    return data

#------DELETE FIT ACTIVITY ENDPOINT-------------------------------------------------------------------

@app.delete('/delete_fit_activity/{activity_id}', summary="Delete fit activity", tags=["Activities"], status_code=status.HTTP_202_ACCEPTED)
def delete_activity(activity_id: int, athlete: UserOut = Depends(get_current_user)):
    with Database() as db:
        db.execute("DELETE FROM public.fit_data WHERE activity_id = %s AND athlete_id = %s", (activity_id, athlete['id']))
        db.close

    return {"message": f"Activity {activity_id} deleted successfully"}

#------DELETE CSV ACTIVITY ENDPOINT-------------------------------------------------------------------

@app.delete('/delete_csv_activity/{activity_id}', summary="Delete csv activity", tags=["Activities"], status_code=status.HTTP_202_ACCEPTED)   
def delete_activity(activity_id: int, athlete: UserOut = Depends(get_current_user)):
    with Database() as db:
        db.execute("DELETE FROM public.csv_data WHERE activity_id = %s AND athlete_id = %s", (activity_id, athlete['id']))
        db.close

    return {"message": f"Activity {activity_id} deleted successfully"}


#------UPLOAD CSV FILE ENDPOINT-------------------------------------------------------------------

@app.post('/upload_csv', summary="Upload csv file", tags=["Activities"], status_code=status.HTTP_201_CREATED)
def upload_file(file: UploadFile = File(...), athlete: UserOut = Depends(get_current_user)):
    df = pd.read_csv(file.file,sep=",")
    #replace NaN with null
    df = df.where(pd.notnull(df), None)
    # If file is missing core and skin temperature columns , add core and skin temperature columns with None values.
    if len(df.columns) < 10:
        df.insert(7, "core_temperature", None)
        df.insert(8, "skin_temperature", None)
    # insert athlete id to df
    df.insert(10, "athlete_id", athlete['id'])

    # Convert first activity timestamp value to hash value to use as a unique activity id
    activity_id = hash(str(df.loc[0, "datetime"])) & ((1 << 32) - 1)
     # insert activity id to df
    df.insert(11, "activity_id", activity_id)

    # close file
    file.file.close()

    # Copy df values to db. Using COPY rather than INSERT query to improve performance on high latency connections.
    # convert dataframe to CSV-formatted byte string in memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, header=False, sep=",")
    csv_buffer.seek(0)
    # copy the CSV-formatted byte string to the database
    try:
        with Database() as db:
            db.copy_expert('''
                           COPY public.csv_data (datetime, latitude, longitude, speed, elevation, heartrate, cadence, 
                                                 core_temperature, skin_temperature, stride_length, athlete_id, activity_id
                                                 ) 
                           FROM STDIN WITH CSV;
                           ''', csv_buffer)       
    except errors.UniqueViolation:
        return {"message": f"CSV file {file.filename} already uploaded"} 
    else:
        return {"message": f"CSV file {file.filename} uploaded successfully"}
    finally:
        if db:
            db.close

#------UPLOAD FIT FILE ENDPOINT----------------------------------------------------------------

@app.post('/upload_fit', summary="Upload fit file", tags=["Activities"], status_code=status.HTTP_201_CREATED)
def upload_fit_file(file: UploadFile = File(...), athlete: UserOut = Depends(get_current_user)):

    # Uploaded FIT file
    fit_file = file.file
     
    # Desired fields to extract from the FIT file
    fields_to_extract = [
                         "timestamp", "position_lat", "position_long", "enhanced_speed", "enhanced_altitude", 
                         "heart_rate", "cadence", "Power","core_temperature", "skin_temperature", "step_length"
                        ]

    # Parse the FIT file and create a pandas DataFrame
    df = parse_fit_file(fit_file, fields_to_extract)

    # Convert first activity timestamp value to hash value to use as a unique activity id
    activity_id = hash(str(df.loc[0, 'timestamp'])) & ((1 << 32) - 1)
    # replace NaN with null
    df = df.where(pd.notnull(df), None)
    # insert athlete id to df
    df.insert(11, "athlete_id", athlete['id'])
    # insert activity id to df
    df.insert(12, "activity_id", activity_id)

    # convert longitude and latitude columns from semicircles to degrees
    df['position_lat'] = df['position_lat'] * 180./ 2**31
    df['position_long'] = df['position_long'] * 180./ 2**31

    file.file.close()

    # Copy df values to db. Using COPY rather than INSERT query to improve performance on high latency connections.
    # convert dataframe to CSV-formatted byte string in memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, header=False, sep=",")
    csv_buffer.seek(0)
    # copy the CSV-formatted byte string to the database
    try:
        with Database() as db:
            db.copy_expert('''
                           COPY public.fit_data (datetime, latitude, longitude, speed, elevation, heartrate, cadence, power, 
                                                 core_temperature, skin_temperature, stride_length, athlete_id, activity_id
                                                 ) 
                           FROM STDIN WITH CSV;
                           ''', csv_buffer)       
    except errors.UniqueViolation:
        return {"message": f"Fit file {file.filename} already uploaded"} 
    else:
        return {"message": f"Fit file {file.filename} uploaded successfully"}
    finally:
        if db:
            db.close

#------BULK UPLOAD CSV FILES IN A ZIPPED FOLDER ENDPOINT------------------------

@app.post('/bulk_upload_csv', summary="Upload csv files in a zipped folder", tags=["Activities"], status_code=status.HTTP_201_CREATED)
def upload_csv_zip_file(file: UploadFile = File(...), athlete: UserOut = Depends(get_current_user)):
    files_skipped = []
    files_uploaded = []
    # Read the zipped folder
    with ZipFile(BytesIO(file.file.read()), 'r') as zip:
        # Loop through each file in the zipped folder
        for csv_file in zip.namelist():
            # Read the csv file
            df = pd.read_csv(zip.open(csv_file),sep=",")
            #replace NaN with null
            df = df.where(pd.notnull(df), None)
            # If file is missing core and skin temperature columns , add core and skin temperature columns with None values.
            if len(df.columns) < 10:
                df.insert(7, "core_temperature", None)
                df.insert(8, "skin_temperature", None)
            # insert athlete id to df
            df.insert(10, "athlete_id", athlete['id'])

            # Convert first activity timestamp value to hash value to use as a unique activity id
            activity_id = hash(str(df.loc[0, "datetime"])) & ((1 << 32) - 1)
            # insert activity id to df
            df.insert(11, "activity_id", activity_id)

            # convert dataframe to CSV-formatted byte string in memory
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False, header=False, sep=",")
            csv_buffer.seek(0)
            # copy the CSV-formatted byte string to the database
            try:
                with Database() as db:
                    db.copy_expert('''
                                   COPY public.csv_data (datetime, latitude, longitude, speed, elevation, heartrate, cadence, 
                                                         core_temperature, skin_temperature, stride_length, athlete_id, activity_id
                                                         ) 
                                   FROM STDIN WITH CSV;
                                   ''', csv_buffer)       
            except errors.UniqueViolation:
                files_skipped.append(csv_file)
            else:
                files_uploaded.append(csv_file)
            finally:
                if db:
                    db.close

    # clean up
    file.file.close()
    zip.close()
  
    return {"message": f"{len(files_uploaded)} csv files uploaded to DB", 
        "Files already in DB and skipped": files_skipped, 
        "Files uploaded": files_uploaded}

#------BULK UPLOAD FIT FILES IN A ZIPPED FOLDER ENDPOINT-----------------------------

@app.post('/bulk_upload_fit', summary="Upload fit files in a zipped folder", tags=["Activities"], status_code=status.HTTP_201_CREATED)
def upload_fit_zip_file(file: UploadFile = File(...), athlete: UserOut = Depends(get_current_user)):
    files_skipped = []
    files_uploaded = []
    # Desired fields to extract from the FIT file
    fields_to_extract = [
                         "timestamp", "position_lat", "position_long", "enhanced_speed", "enhanced_altitude", 
                         "heart_rate", "cadence", "Power","core_temperature", "skin_temperature", "step_length"
                        ]

    # Read the zipped folder
    with ZipFile(BytesIO(file.file.read()), 'r') as zip:
        # Loop through each file in the zipped folder
        for fit_file in zip.namelist():
            # Parse the FIT file and create a pandas DataFrame
            df = parse_fit_file(zip.open(fit_file), fields_to_extract)
            # Convert first activity timestamp value to hash value to use as a unique activity id
            activity_id = hash(str(df.loc[0, 'timestamp'])) & ((1 << 32) - 1)
            # replace NaN with null
            df = df.where(pd.notnull(df), None)
            # insert athlete id to df
            df.insert(11, "athlete_id", athlete['id'])
            # insert activity id to df
            df.insert(12, "activity_id", activity_id)
            # convert longitude and latitude columns from semicircles to degrees
            df['position_lat'] = df['position_lat'] * 180./ 2**31
            df['position_long'] = df['position_long'] * 180./ 2**31

            # Copy df values to db. Using COPY rather than INSERT query to improve performance on high latency connections.
            # convert dataframe to CSV-formatted byte string in memory
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False, header=False, sep=",")
            csv_buffer.seek(0)
            # copy the CSV-formatted byte string to the database
            try:
                with Database() as db:
                    db.copy_expert('''
                                   COPY public.fit_data (datetime, latitude, longitude, speed, elevation, heartrate, cadence, power, 
                                                         core_temperature, skin_temperature, stride_length, athlete_id, activity_id
                                                         ) 
                                   FROM STDIN WITH CSV;
                                   ''', csv_buffer)       
            except errors.UniqueViolation:
                files_skipped.append(fit_file)
            else:
                files_uploaded.append(fit_file)
            finally:
                if db:
                    db.close

    # clean up
    file.file.close()
    zip.close()
  
    return {"message": f"{len(files_uploaded)} fit files uploaded to DB", 
        "Files already in DB and skipped": files_skipped, 
        "Files uploaded": files_uploaded}

#------GET CSV DATA ENDPOINT-------------------------------------------------------------------

# get activity data from csv_data table where activity_id = activity_id
@app.get('/csv_activity/{activity_id}', summary="Get activity data from csv_data table", tags=["Activities"])
def get_csv_data(activity_id: int, athlete: UserOut = Depends(get_current_user)):
    with Database() as db:
        data = db.query_fetchall('''
                                 SELECT datetime,latitude,longitude,speed,elevation,heartrate,cadence,core_temperature,skin_temperature,stride_length 
                                 FROM public.csv_data 
                                 WHERE athlete_id = %s AND activity_id = %s"
                                 ''', (athlete['id'],activity_id))
        db.close

    '''
    # return data as a list of dicts - ["datetime":{1},"datetime":{2},"datetime":{3}"]
    data = [{"datetime": row[0], 
             "latitude": row[1], 
             "longitude": row[2], 
             "speed": row[3], 
             "elevation": row[4], 
             "heartrate": row[5], 
             "cadence": row[6], 
             "core_temperature": row[7], 
             "skin_temperature": row[8], 
             "stride_length": row[9]} for row in data]
    '''
    # return data as dict of lists - "datetime":[1,2,3]
    data = {"datetime": [row[0] for row in data],
            "latitude": [row[1] for row in data], 
            "longitude": [row[2] for row in data],
            "speed": [row[3] for row in data],
            "elevation": [row[4] for row in data],
            "heartrate": [row[5] for row in data],
            "cadence": [row[6] for row in data],
            "core_temperature": [row[7] for row in data],
            "skin_temperature": [row[8] for row in data],
            "stride_length": [row[9] for row in data]
            }
    
    return data

#------GET FIT DATA ENDPOINT-------------------------------------------------------------------

# get activity data from fit_data table where activity_id = activity_id
@app.get('/fit_activity/{activity_id}', summary="Get activity data from fit_data table", tags=["Activities"])
def get_fit_data(activity_id: int, athlete: UserOut = Depends(get_current_user)):
    with Database() as db:
        data = db.query_fetchall('''
                                 SELECT datetime,latitude,longitude,speed,elevation,heartrate,cadence,
                                        power,core_temperature,skin_temperature,stride_length 
                                 FROM public.fit_data 
                                 WHERE athlete_id = %s AND activity_id = %s
                                 ''', (athlete['id'],activity_id))
        db.close

    '''
    # return data as a list of dicts - ["datetime":{1},"datetime":{2},"datetime":{3}"]
    data = [{"datetime": row[0], 
             "latitude": row[1], 
             "longitude": row[2], 
             "speed": row[3], 
             "elevation": row[4], 
             "heartrate": row[5], 
             "cadence": row[6], 
             "power": row[7], 
             "core_temperature": row[8], 
             "skin_temperature": row[9], 
             "stride_length": row[10]} for row in data]
    '''
    # return data as dict of lists - "datetime":[1,2,3]
    data = {"datetime": [row[0] for row in data],
            "latitude": [row[1] for row in data], 
            "longitude": [row[2] for row in data],
            "speed": [row[3] for row in data],
            "elevation": [row[4] for row in data],
            "heartrate": [row[5] for row in data],
            "cadence": [row[6] for row in data],
            "power": [row[7] for row in data],
            "core_temperature": [row[8] for row in data],
            "skin_temperature": [row[9] for row in data],
            "stride_length": [row[10] for row in data]
            }
    
    return data

#------GET MEAN-MAX CURVE FOR THE SELECTED COLUMN AND TABLE-----------------------------------------------

# Define a /mean-max endpoint to return the mean-max values for the given parameters.
@app.get('/activity_mean_max', summary="Get mean-max values for the given parameters", tags=["Analysis"])
def get_mean_max(column: str, table: str, activity_id: int, athlete: UserOut = Depends(get_current_user)):
    # check that the column is valid
    if column not in ['heartrate', 'speed', 'cadence', 'power', 'core_temperature', 'skin_temperature', 'stride_length']:
        raise HTTPException(status_code=400, detail="Invalid column name")
    # check that the table is valid
    if table not in ['csv_data', 'fit_data']:
        raise HTTPException(status_code=400, detail="Invalid table name")
    # check that the activity_id is valid
    with Database() as db:
        activity_id_check = db.query_fetchone('''
                                             SELECT activity_id 
                                             FROM public.{}
                                             WHERE athlete_id = %s AND activity_id = %s
                                             '''.format(table), (athlete['id'],activity_id))
        db.close
    if not activity_id_check:
        raise HTTPException(status_code=400, detail="Invalid activity_id")
    
    # get the column values for the given parameters
    with Database() as db:
        data = db.query_fetchall('''
                                 SELECT {} 
                                 FROM public.{} 
                                 WHERE athlete_id = %s AND activity_id = %s
                                 '''.format(column, table), (athlete['id'],activity_id))
        db.close

    # Store the results in the dataframe
    df = pd.DataFrame(data, columns=[column])

    # Drop NaN values
    df = df.dropna()

    # Calculate the maximum rolling average value for each window size
    # The code calculates the rolling mean and maximum value of response_df for different window sizes, and stores the maximum value for each window size in a list.
    mean_max = [df.rolling(d).mean().max() for d in range(len(df))]

    # Create a dictionary with the window size as key and the maximum rolling average of the given column as value
    mean_max_dict = dict(zip(range(len(df)), mean_max))
    # Remove the NaN values from the dictionary. Typically, the first key:value in the dictionary is NaN.
    mean_max_dict = {k: v for k, v in mean_max_dict.items() if str(v) != 'nan'}

    # Round float values to 3 decimal places and exclude the pair if not finite
    mean_max_dict = {k: round(v, 3) for k, v in mean_max_dict.items() if math.isfinite(v)}

    # return data as dict of lists - "duration":[1,2,3],mean_max":[1,2,3]
    data = {"duration": list(mean_max_dict.keys()),
            column: [item[column] for item in list(mean_max_dict.values())]
            }
    
    return data

#------GET A RESPONSE FROM PANDAS DATAFRAME USING PROMT IN NATURAL LANGUAGE----------------------------------------

@app.get('/nat_lang_query', summary="Generate answers using OpenAI API", tags=["Experimental"])
def get_pandas_response(prompt:str, table: str, activity_id: int, model: str = 'text-davinci-003', athlete: UserOut = Depends(get_current_user)):

    # check that the table is valid
    if table not in ['csv_data', 'fit_data']:
        raise HTTPException(status_code=400, detail="Invalid table name")
    # check that the activity_id is valid
    with Database() as db:
        activity_id_check = db.query_fetchone('''
                                             SELECT activity_id 
                                             FROM public.{}
                                             WHERE athlete_id = %s AND activity_id = %s
                                             
                                             '''.format(table), (athlete['id'],activity_id))
        db.close
    if not activity_id_check:
        raise HTTPException(status_code=400, detail="Invalid activity_id")
    
    # get the column values for the given parameters
    with Database() as db:
        data = db.query_fetchall('''
                                 SELECT datetime, latitude, longitude, speed, elevation, 
                                 heartrate, cadence, power, core_temperature, skin_temperature, stride_length 
                                 FROM public.{} 
                                 WHERE athlete_id = %s AND activity_id = %s
                                 '''.format(table), (athlete['id'],activity_id))
        db.close

    # Store the results in the dataframe
    df = pd.DataFrame(data, columns=['datetime', 'latitude', 'longitude', 'speed', 'elevation', 'heartrate', 'cadence', 'power', 'core_temperature', 'skin_temperature', 'stride_length'])
    # Convert datetime column to datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Drop NaN values
    df = df.dropna()

    # Here is where the magic happens :-). The dataframe is passed to the agent, which is then used to generate a response.
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0, model=model ,openai_api_key=api_key, max_retries=2), df, verbose=True)

    response = agent.run(prompt)

    return response




