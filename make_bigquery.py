from google.cloud import bigquery

def create_table(table_id):

    # [START bigquery_create_table]
    # Construct a BigQuery client object.
    client = bigquery.Client()

    schema = [
        bigquery.SchemaField("tweet", "STRING", mode="REQUIRED"),
		bigquery.SchemaField("proc_tweet", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("tweet_tp", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("exp_score", "NUMERIC", mode="REQUIRED"),
		bigquery.SchemaField("loc", "STRING", mode="REQUIRED"),
		bigquery.SchemaField("reg_dt", "TIMESTAMP", mode="REQUIRED"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table)  # Make an API request.
    print(
        "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
    )
    # [END bigquery_create_table]

# TODO(developer): Set table_id to the ID of the table to create.
table_id = "engineering123.twitter_project.sent_anl_rst"
create_table(table_id)

"""
from google.cloud import bigquery

def create_table(table_id):

    # [START bigquery_create_table]
    # Construct a BigQuery client object.
    client = bigquery.Client()

    schema = [
        bigquery.SchemaField("remote_addr", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("timelocal", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("request_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("body_bytes_sent", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("http_referer", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("http_user_agent", "STRING", mode="REQUIRED"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table)  # Make an API request.
    print(
        "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
    )
    # [END bigquery_create_table]

# TODO(developer): Set table_id to the ID of the table to create.
table_id = "engineering123.fakeds.realdata"
create_table(table_id)
"""