from datetime import datetime, timedelta, date
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import numpy as np
import asyncio
import os
import nest_asyncio
nest_asyncio.apply()

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 13),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'tiktok_trending_scraper',
    default_args=default_args,
    description='Daily TikTok trending videos scraping',
    schedule_interval='@daily',
    catchup=False,
    tags=['tiktok', 'social-media', 'trending'],
)

def check_tiktok_dependencies():
    """Check if TikTokApi is available"""
    try:
        from TikTokApi import TikTokApi
        print("TikTokApi is available")
        return True
    except ImportError:
        print("TikTokApi not found. Please install with: pip install TikTokApi")
        return False

async def scrape_tiktok_trending_videos(ms_token, count=50):
    """Scrape TikTok trending videos"""
    print(f"Starting TikTok trending videos scraping")
    
    try:
        from TikTokApi import TikTokApi
        
        videos_data = []
        async with TikTokApi() as api:
            await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, browser=os.getenv("TIKTOK_BROWSER", "chromium"))
            
            video_count = 0
            async for video in api.trending.videos(count=count):
                try:
                    video_dict = video.as_dict
                    
                    # Extract relevant data based on the sample structure provided
                    video_data = {
                        'video_id': video_dict.get('id', ''),
                        'caption': video_dict.get('desc', ''),
                        'username': video_dict.get('author', {}).get('uniqueId', ''),
                        'author_name': video_dict.get('author', {}).get('nickname', ''),
                        'like_count': video_dict.get('stats', {}).get('diggCount', 0),
                        'comment_count': video_dict.get('stats', {}).get('commentCount', 0),
                        'share_count': video_dict.get('stats', {}).get('shareCount', 0),
                        'play_count': video_dict.get('stats', {}).get('playCount', 0),
                        'create_time': video_dict.get('createTime', ''),
                        'video_duration': video_dict.get('video', {}).get('duration', 0),
                        'music_title': video_dict.get('music', {}).get('title', ''),
                        'music_author': video_dict.get('music', {}).get('authorName', '')
                    }
                    
                    videos_data.append(video_data)
                    video_count += 1
                    print(f"Processed video {video_count}: {video_data['username']} - {video_data['caption'][:50]}...")
                    
                except Exception as video_error:
                    print(f"Error processing individual video: {str(video_error)}")
                    continue
        
        print(f"Successfully scraped {len(videos_data)} TikTok videos")
        return videos_data, None
        
    except ImportError:
        error_msg = "TikTokApi not installed. Please install with: pip install TikTokApi"
        print(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Error scraping TikTok videos: {str(e)}"
        print(error_msg)
        return None, error_msg

def scrape_tiktok_data_sync(ms_token, count=5):
    """Synchronous wrapper for TikTok scraping"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's already a running event loop, create a new one
            import threading
            import concurrent.futures
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(scrape_tiktok_trending_videos(ms_token, count))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=300)  # 5 minute timeout
        else:
            return loop.run_until_complete(scrape_tiktok_trending_videos(ms_token, count))
    except Exception as e:
        return None, f"Error in synchronous TikTok scraper: {str(e)}"

def process_tiktok_data(videos_data, scrape_date):
    """Process TikTok videos data and prepare for database insertion"""
    try:
        if not videos_data:
            return None, "No TikTok videos data provided"
        
        print(f"Processing {len(videos_data)} TikTok videos")
        
        # Convert to DataFrame
        df = pd.DataFrame(videos_data)
        
        if df.empty:
            return None, "TikTok videos data is empty"
        
        # Add scrape date
        df['scrape_date'] = scrape_date
        
        # Ensure all required columns are present
        required_columns = [
            'video_id', 'caption', 'username', 'author_name', 'like_count',
            'comment_count', 'share_count', 'play_count', 'create_time',
            'video_duration', 'music_title', 'music_author', 'scrape_date'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        df_clean = df[required_columns].copy()
        
        # Clean and validate video IDs
        df_clean['video_id'] = df_clean['video_id'].astype(str)
        missing_id_mask = (df_clean['video_id'].isnull()) | (df_clean['video_id'] == 'None') | (df_clean['video_id'] == '') | (df_clean['video_id'] == 'nan')
        
        if missing_id_mask.sum() > 0:
            print(f"Found {missing_id_mask.sum()} videos with missing IDs, generating synthetic IDs")
            import hashlib
            for idx in df_clean[missing_id_mask].index:
                content = str(df_clean.loc[idx, 'caption']) + str(df_clean.loc[idx, 'username']) + str(df_clean.loc[idx, 'create_time'])
                synthetic_id = f"tiktok_{scrape_date}_{hashlib.md5(content.encode()).hexdigest()[:12]}"
                df_clean.loc[idx, 'video_id'] = synthetic_id
        
        # Remove rows with invalid IDs
        before_id_filter = len(df_clean)
        df_clean = df_clean[~((df_clean['video_id'].isnull()) | (df_clean['video_id'] == 'None') | (df_clean['video_id'] == '') | (df_clean['video_id'] == 'nan'))]
        after_id_filter = len(df_clean)
        
        if before_id_filter != after_id_filter:
            print(f"Removed {before_id_filter - after_id_filter} videos with invalid IDs")
        
        if df_clean.empty:
            return None, "No valid videos remaining after ID validation"
        
        # Clean numeric columns
        numeric_columns = ['like_count', 'comment_count', 'share_count', 'play_count', 'video_duration']
        for col in numeric_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
        
        # Convert create_time to readable format if it's a timestamp
        try:
            # Handle both string and numeric timestamps
            df_clean['created_at'] = pd.to_datetime(df_clean['create_time'], unit='s', errors='coerce')
            # If conversion failed, try to convert as string
            failed_mask = df_clean['created_at'].isnull()
            if failed_mask.any():
                df_clean.loc[failed_mask, 'created_at'] = pd.to_datetime(df_clean.loc[failed_mask, 'create_time'], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not convert timestamps: {e}")
            df_clean['created_at'] = None
        
        # Remove duplicates based on video_id
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['video_id'], keep='first')
        final_count = len(df_clean)
        duplicates_removed = initial_count - final_count
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate videos")
        
        # Select only the columns that match the database schema
        final_columns = [
            'video_id', 'caption', 'username', 'author_name', 'like_count',
            'comment_count', 'share_count', 'play_count', 'create_time',
            'created_at', 'video_duration', 'music_title', 'music_author', 'scrape_date'
        ]
        
        # Ensure all final columns exist
        for col in final_columns:
            if col not in df_clean.columns:
                df_clean[col] = None
        
        df_final = df_clean[final_columns].copy()
        
        print(f"Processed {final_count} unique TikTok videos")
        return df_final, None
        
    except Exception as e:
        error_msg = f"Error processing TikTok videos data: {str(e)}"
        print(error_msg)
        return None, error_msg

def scrape_tiktok_trending(**context):
    """Main function: Scrape TikTok trending videos for current date and save to PostgreSQL"""
    DB_CONFIG = {
        'host': 'localhost',
        'database': 'appdb',
        'user': 'appuser',
        'password': '*****************'
    }
    
    TIKTOK_CONFIG = {
        'ms_token': '*********************************************************************',
        'video_count': 10
    }
    
    date_str = context['ds']
    print(f"Starting TikTok trending videos scraping for execution date: {date_str}")
    
    try:
        engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")
        print("Database connection established")
    except Exception as e:
        raise Exception(f"Failed to connect to database: {str(e)}")
    
    try:
        if not check_tiktok_dependencies():
            raise Exception("TikTokApi is not available. Please install with: pip install TikTokApi")
        
        os.makedirs('tiktok-data', exist_ok=True)
        
        videos_data, scrape_error = scrape_tiktok_data_sync(
            TIKTOK_CONFIG['ms_token'], 
            TIKTOK_CONFIG['video_count']
        )
        
        if scrape_error:
            raise Exception(f"TikTok scraping failed: {scrape_error}")
        
        if not videos_data:
            raise Exception("No TikTok videos data was generated")
        
        processed_df, process_error = process_tiktok_data(videos_data, date_str)
        
        if process_error:
            raise Exception(f"Data processing failed: {process_error}")
        
        if processed_df is None or processed_df.empty:
            raise Exception("No valid TikTok data was processed")
        
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS tiktok_data (
                "video_id" VARCHAR(100) PRIMARY KEY,
                "caption" TEXT,
                "username" VARCHAR(100),
                "author_name" VARCHAR(200),
                "like_count" INTEGER DEFAULT 0,
                "comment_count" INTEGER DEFAULT 0,
                "share_count" INTEGER DEFAULT 0,
                "play_count" INTEGER DEFAULT 0,
                "create_time" VARCHAR(50),
                "created_at" TIMESTAMP,
                "video_duration" INTEGER DEFAULT 0,
                "music_title" TEXT,
                "music_author" VARCHAR(200),
                "scrape_date" VARCHAR(20),
                "created_at_db" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            engine.execute(text(create_table_sql))
            print("TikTok data table created/verified")
            
            create_error_table_sql = """
            CREATE TABLE IF NOT EXISTS tiktok_errors (
                id SERIAL PRIMARY KEY,
                scrape_date VARCHAR(20),
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            engine.execute(text(create_error_table_sql))
            
        except Exception as table_error:
            print(f"Table setup error: {str(table_error)}")
        
        try:
            temp_table = f"temp_tiktok_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            processed_df.to_sql(temp_table, engine, if_exists='replace', index=False)
            
            insert_sql = f"""
            INSERT INTO tiktok_data (
                video_id, caption, username, author_name, like_count,
                comment_count, share_count, play_count, create_time,
                created_at, video_duration, music_title, music_author, scrape_date
            )
            SELECT 
                video_id, caption, username, author_name, like_count,
                comment_count, share_count, play_count, create_time,
                created_at, video_duration, music_title, music_author, scrape_date
            FROM {temp_table}
            ON CONFLICT (video_id) DO UPDATE SET
                caption = EXCLUDED.caption,
                like_count = EXCLUDED.like_count,
                comment_count = EXCLUDED.comment_count,
                share_count = EXCLUDED.share_count,
                play_count = EXCLUDED.play_count,
                scrape_date = EXCLUDED.scrape_date;
            """
            result = engine.execute(text(insert_sql))
            inserted_count = result.rowcount if hasattr(result, 'rowcount') else len(processed_df)
            
            engine.execute(text(f"DROP TABLE {temp_table}"))
            
            print(f"TikTok data saved to PostgreSQL database")
            print(f"Total records processed: {len(processed_df)}")
            print(f"Records inserted/updated: {inserted_count}")
            
            return f"Successfully scraped TikTok trending videos. Processed: {len(processed_df)}, Inserted/Updated: {inserted_count} records"
            
        except Exception as e:
            error_msg = f"Failed to save data to database: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
        
    except Exception as main_error:
        error_summary = str(main_error)
        print(f"Main error: {error_summary}")
        
        try:
            error_df = pd.DataFrame({
                'scrape_date': [date_str],
                'error_message': [error_summary],
                'created_at': [datetime.now()]
            })
            error_df.to_sql('tiktok_errors', engine, if_exists='append', index=False)
            print("Error logged to database")
        except Exception as db_error:
            print(f"Failed to log error to database: {str(db_error)}")
        
        raise Exception(error_summary)
    
    finally:
        engine.dispose()

scrape_task = PythonOperator(
    task_id='scrape_tiktok_trending',
    python_callable=scrape_tiktok_trending,
    dag=dag,
)

scrape_task
