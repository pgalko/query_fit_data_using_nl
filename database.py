import psycopg2

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/database_name" # Local instance connection string
DATABASE_URL_SUPABASE = "postgresql://postgres:password@db.database.supabase.co:5432/postgres" # Supabase connection string
CREATE_SCHEMA = '''
SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

CREATE TABLE IF NOT EXISTS public.athlete (
    id bigint NOT NULL,
    username character varying,
    password character varying
);

CREATE SEQUENCE IF NOT EXISTS public.athlete_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.athlete_id_seq OWNED BY public.athlete.id;

CREATE TABLE IF NOT EXISTS public.csv_data (
    id integer NOT NULL,
    athlete_id integer,
    activity_id bigint,
    datetime character varying,
    latitude real,
    longitude real,
    speed real,
    elevation real,
    heartrate real,
    cadence real,
    core_temperature real,
    skin_temperature real,
    stride_length real
);

CREATE SEQUENCE IF NOT EXISTS public.csv_data_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.csv_data_id_seq OWNED BY public.csv_data.id;

CREATE TABLE IF NOT EXISTS public.fit_data (
    id integer NOT NULL,
    athlete_id integer,
    activity_id bigint,
    datetime character varying,
    latitude real,
    longitude real,
    speed real,
    elevation real,
    heartrate real,
    cadence real,
    power real,
    core_temperature real,
    skin_temperature real,
    stride_length real
);

CREATE SEQUENCE IF NOT EXISTS public.fit_data_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.fit_data_id_seq OWNED BY public.csv_data.id;

ALTER TABLE ONLY public.athlete ALTER COLUMN id SET DEFAULT nextval('public.athlete_id_seq'::regclass);

ALTER TABLE ONLY public.csv_data ALTER COLUMN id SET DEFAULT nextval('public.csv_data_id_seq'::regclass);

ALTER TABLE ONLY public.fit_data ALTER COLUMN id SET DEFAULT nextval('public.fit_data_id_seq'::regclass);

ALTER TABLE ONLY public.athlete
    ADD CONSTRAINT athlete_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.csv_data
    ADD CONSTRAINT csv_data_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.fit_data
    ADD CONSTRAINT fit_data_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.csv_data
    ADD CONSTRAINT unq_csv_datetime UNIQUE (athlete_id, datetime);

ALTER TABLE ONLY public.fit_data
    ADD CONSTRAINT unq_fit_datetime UNIQUE (athlete_id, datetime);
	
ALTER TABLE ONLY public.csv_data
    ADD CONSTRAINT fk_csv_athlete_id FOREIGN KEY (athlete_id) REFERENCES public.athlete(id) NOT VALID;

ALTER TABLE ONLY public.fit_data
    ADD CONSTRAINT fk_fit_athlete_id FOREIGN KEY (athlete_id) REFERENCES public.athlete(id) NOT VALID;
'''

class Database:   
    def __init__(self, name=DATABASE_URL_SUPABASE): # Replace with DATABESE_URL if using local instance
        self._conn = psycopg2.connect(name)
        self._cursor = self._conn.cursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def connection(self):
        return self._conn

    @property
    def cursor(self):
        return self._cursor

    def create_schema(self):
        self._cursor.execute("SELECT * FROM information_schema.tables WHERE table_name=%s", ('athlete',))
        table_exists = bool(self._cursor.rowcount)
        if not table_exists:
            self._cursor.execute(CREATE_SCHEMA)

    def commit(self):
        self.connection.commit()

    def close(self, commit=True):
        if commit:
            self.commit()
        self.connection.close()

    def execute(self, sql, params=None):
        self.cursor.execute(sql, params or ())

    def copy_expert(self, sql, file):
        self.cursor.copy_expert(sql, file)

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchone(self):
        return self.cursor.fetchone()

    def query_fetchall(self, sql, params=None):
        self.cursor.execute(sql, params or ())
        return self.fetchall()
    
    def query_fetchone(self, sql, params=None):
        self.cursor.execute(sql, params or ())
        return self.fetchone()
        
