from datetime import datetime, timedelta, date
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import yfinance as yf
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import numpy as np

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_data_scraper',
    default_args=default_args,
    description='Daily stock data scraping from Yahoo Finance',
    schedule_interval='@daily',
    catchup=False,
    tags=['finance', 'stocks', 'yfinance'],
)

def check_first_run(engine):
    """Check if this is the first run by checking if stock_data table exists and has data"""
    try:
        result = engine.execute(text("SELECT COUNT(*) FROM stock_data LIMIT 1"))
        count = result.scalar()
        return count == 0
    except Exception:
        # Table doesn't exist, so this is definitely first run
        return True

def get_date_range(start_date_str, end_date_str):
    """Generate list of dates between start and end date"""
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    
    return date_list

def scrape_historical_data(symbols, start_date, end_date, engine):
    """Scrape historical data for all symbols from start_date to end_date"""
    print(f"Starting HISTORICAL scraping from {start_date} to {end_date}")
    
    all_data = []
    errors = []
    
    for symbol in symbols:
        try:
            print(f"Scraping historical data for {symbol}...")
            stock = yf.Ticker(symbol)
            
            # Get historical data from start_date to end_date
            hist = stock.history(start=start_date, end=end_date)
            
            if not hist.empty:
                hist.reset_index(inplace=True)
                hist['Symbol'] = symbol
                hist['Scrape_Date'] = end_date  # Use end_date as scrape date
                all_data.append(hist)
                print(f"Successfully scraped {len(hist)} historical records for {symbol}")
            else:
                error_msg = f"No historical data found for {symbol}"
                print(error_msg)
                errors.append(error_msg)
                
        except Exception as e:
            error_msg = f"Error scraping historical data for {symbol}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    
    return all_data, errors

def scrape_single_day_data(symbols, target_date, engine):
    """Scrape data for a single day"""
    print(f"Starting SINGLE DAY scraping for date: {target_date}")
    
    all_data = []
    errors = []
    
    for symbol in symbols:
        try:
            print(f"Scraping single day data for {symbol}...")
            stock = yf.Ticker(symbol)
            
            # Get data for the target date (use 5d period and filter)
            hist = stock.history(period="5d")
            
            if not hist.empty:
                # Filter for the target date or get most recent if target date not available
                hist.reset_index(inplace=True)
                hist['Date'] = pd.to_datetime(hist['Date']).dt.date
                
                # Try to get exact date first
                target_data = hist[hist['Date'] == datetime.strptime(target_date, '%Y-%m-%d').date()]
                
                if target_data.empty:
                    # If exact date not found, get the most recent data
                    target_data = hist.tail(1)
                    print(f"Exact date not found for {symbol}, using most recent data")
                
                target_data = target_data.copy()
                target_data['Symbol'] = symbol
                target_data['Scrape_Date'] = target_date
                all_data.append(target_data)
                print(f"Successfully scraped data for {symbol}")
            else:
                error_msg = f"No data found for {symbol} on {target_date}"
                print(error_msg)
                errors.append(error_msg)
                
        except Exception as e:
            error_msg = f"Error scraping {symbol} for {target_date}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    
    return all_data, errors

def scrape_stock_data(**context):
    """
    Main function: Scrape historical data on first run, single day on subsequent runs
    """
    # Database connection parameters
    DB_CONFIG = {
        'host': 'localhost',
        'database': 'appdb',
        'user': 'appuser',
        'password': '*******************'
    }
    
    # List of stock symbols to scrape
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY']
    
    # Get current date
    date_str = context['ds']  # YYYY-MM-DD format
    
    print(f"Starting stock data scraping for execution date: {date_str}")
    
    # Create database connection
    try:
        engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")
        print("Database connection established")
    except Exception as e:
        raise Exception(f"Failed to connect to database: {str(e)}")
    
    try:
        # Check if this is first run
        is_first_run = check_first_run(engine)
        print(f"First run detected: {is_first_run}")
        
        if is_first_run:
            # Historical scraping from 2020-09-12 to today
            start_date = "2020-09-12"
            end_date = datetime.now().strftime('%Y-%m-%d')
            all_data, errors = scrape_historical_data(symbols, start_date, end_date, engine)
        else:
            # Single day scraping
            all_data, errors = scrape_single_day_data(symbols, date_str, engine)
        
        if all_data:
            # Combine all stock data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates based on Date and Symbol
            initial_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['Date', 'Symbol'], keep='first')
            final_count = len(combined_df)
            duplicates_removed = initial_count - final_count
            
            if duplicates_removed > 0:
                print(f"Removed {duplicates_removed} duplicate records")
            
            # Handle table creation and constraint management
            try:
                # First, try to add unique constraint if table exists without it
                try:
                    constraint_sql = """
                    ALTER TABLE stock_data 
                    ADD CONSTRAINT unique_date_symbol UNIQUE ("Date", "Symbol");
                    """
                    engine.execute(text(constraint_sql))
                    print("Added unique constraint to existing table")
                except Exception as constraint_error:
                    # Constraint might already exist or table might not exist
                    print(f"Constraint handling: {str(constraint_error)}")
                
                # Create table with constraint if it doesn't exist
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS stock_data (
                    "Date" DATE,
                    "Open" FLOAT,
                    "High" FLOAT,
                    "Low" FLOAT,
                    "Close" FLOAT,
                    "Volume" BIGINT,
                    "Dividends" FLOAT,
                    "Stock Splits" FLOAT,
                    "Symbol" VARCHAR(10),
                    "Scrape_Date" VARCHAR(20),
                    CONSTRAINT unique_date_symbol UNIQUE("Date", "Symbol")
                );
                """
                engine.execute(text(create_table_sql))
                
            except Exception as table_error:
                print(f"Table setup error: {str(table_error)}")
            
            # Save to PostgreSQL database with conflict handling
            try:
                temp_table = f"temp_stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                combined_df.to_sql(temp_table, engine, if_exists='replace', index=False)
                
                # Check if unique constraint exists before using ON CONFLICT
                constraint_check_sql = """
                SELECT COUNT(*) FROM information_schema.table_constraints 
                WHERE table_name = 'stock_data' 
                AND constraint_type = 'UNIQUE'
                AND constraint_name = 'unique_date_symbol';
                """
                constraint_exists = engine.execute(text(constraint_check_sql)).scalar() > 0
                
                if constraint_exists:
                    # Use ON CONFLICT if constraint exists
                    insert_sql = f"""
                    INSERT INTO stock_data 
                    SELECT * FROM {temp_table}
                    ON CONFLICT ("Date", "Symbol") DO NOTHING;
                    """
                    result = engine.execute(text(insert_sql))
                    inserted_count = result.rowcount if hasattr(result, 'rowcount') else len(combined_df)
                else:
                    # Fallback: Insert and handle duplicates in application logic
                    print("No unique constraint found, using application-level duplicate handling")
                    
                    # Get existing records to avoid duplicates
                    existing_records_sql = f"""
                    SELECT "Date", "Symbol" FROM stock_data 
                    WHERE ("Date", "Symbol") IN (
                        SELECT "Date", "Symbol" FROM {temp_table}
                    );
                    """
                    
                    try:
                        existing_df = pd.read_sql(existing_records_sql, engine)
                        existing_df['exists'] = True
                        
                        # Merge to find new records only
                        merged_df = combined_df.merge(existing_df, on=['Date', 'Symbol'], how='left')
                        new_records_df = merged_df[merged_df['exists'].isnull()].drop('exists', axis=1)
                        
                        if not new_records_df.empty:
                            new_records_df.to_sql('stock_data', engine, if_exists='append', index=False)
                            inserted_count = len(new_records_df)
                        else:
                            inserted_count = 0
                    except Exception as fallback_error:
                        print(f"Fallback method failed: {str(fallback_error)}")
                        # Last resort: just append and let database handle any errors
                        combined_df.to_sql('stock_data', engine, if_exists='append', index=False)
                        inserted_count = len(combined_df)
                
                # Drop temp table
                engine.execute(text(f"DROP TABLE {temp_table}"))
                
                print(f"Stock data saved to PostgreSQL database")
                print(f"Total records processed: {len(combined_df)}")
                print(f"Records actually inserted: {inserted_count}")
                
                # Log any errors to database
                if errors:
                    error_df = pd.DataFrame({
                        'scrape_date': [date_str] * len(errors),
                        'error_message': errors,
                        'created_at': [datetime.now()] * len(errors)
                    })
                    error_df.to_sql('stock_errors', engine, if_exists='append', index=False)
                    print(f"Errors logged to database: {len(errors)} errors")
                
                run_type = "HISTORICAL" if is_first_run else "DAILY"
                return f"Successfully completed {run_type} scraping. Processed: {len(combined_df)}, Inserted: {inserted_count} records"
                
            except Exception as e:
                error_msg = f"Failed to save data to database: {str(e)}"
                print(error_msg)
                raise Exception(error_msg)
            
        else:
            error_summary = f"No stock data was successfully scraped. Errors: {'; '.join(errors)}"
            print(error_summary)
            
            # Log errors to database
            try:
                error_df = pd.DataFrame({
                    'scrape_date': [date_str],
                    'error_message': [error_summary],
                    'created_at': [datetime.now()]
                })
                error_df.to_sql('stock_errors', engine, if_exists='append', index=False)
            except Exception as db_error:
                print(f"Failed to log errors to database: {str(db_error)}")
            
            raise Exception(error_summary)
    
    finally:
        engine.dispose()

scrape_task = PythonOperator(
    task_id='scrape_stock_data',
    python_callable=scrape_stock_data,
    dag=dag,
)

scrape_task