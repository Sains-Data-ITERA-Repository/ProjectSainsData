"""
Silver Layer ETL: Production Data Cleaning Script

This script processes production data from the bronze layer, performs cleaning
and validation, and writes the cleaned data to the silver layer with proper
partitioning.
"""

import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pyspark.sql.functions import col
from pyspark.sql.functions import round as spark_round
from pyspark.sql.types import DoubleType, IntegerType

from config.settings import FILE_CONFIG, HDFS_CONFIG, HIVE_CONFIG, QUALITY_CONFIG
from utils.data_quality import (
    calculate_data_quality_score,
    clean_invalid_markers,
    generate_quality_report,
    standardize_region_names,
    validate_bronze_data,
)
from utils.spark_utils import (
    SparkSessionManager,
    clean_text_column,
    create_hive_table,
    read_csv_with_file_config,
    validate_data_quality,
    write_parquet_to_hdfs,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_bronze_data(spark):
    """Load production data from bronze layer"""
    logger.info("Loading production data from bronze layer")

    input_path = f"{str(HDFS_CONFIG['data_paths']['bronze'])}/produksi_pangan.csv"
    filename = "produksi_pangan.csv"

    df = read_csv_with_file_config(
        spark,
        input_path,
        filename,
        header=bool(FILE_CONFIG["header"]),
        infer_schema=bool(FILE_CONFIG["infer_schema"]),
    )

    logger.info(f"Loaded {df.count()} rows from bronze layer")
    return df


def validate_input_data(df):
    """Validate input data quality"""
    logger.info("Validating input data quality")

    validation_results = validate_bronze_data(df, "produksi_pangan")
    quality_score = calculate_data_quality_score(validation_results)

    logger.info(f"Input data quality score: {quality_score}/100")

    if quality_score < 60:
        logger.warning("Input data quality is below acceptable threshold")
        print(generate_quality_report(validation_results))

    return validation_results


def convert_data_types(df):
    """Convert columns to appropriate data types"""
    return (df
            .withColumn("tahun", col("tahun").cast(IntegerType()))
            .withColumn("produksi", col("produksi").cast(DoubleType()))
            .withColumn("luas_panen", col("luas_panen").cast(DoubleType()))
            .withColumn("produktivitas", col("produktivitas").cast(DoubleType())))


def calculate_derived_metrics(df):
    """Calculate derived productivity metrics"""
    return df.withColumn(
        "produktivitas_per_ha",
        spark_round(col("produksi") / col("luas_panen"), 4)
    )


def filter_valid_records(df):
    """Filter out invalid records"""
    return df.filter(
        col("produksi").isNotNull()
        & (col("produksi") > 0)
        & col("luas_panen").isNotNull()
        & (col("luas_panen") > 0)
        & col("tahun").isNotNull()
        & col("kabupaten_kota").isNotNull()
        & (col("tahun") >= 2000)
        & (col("tahun") <= 2030)
        & (col("produksi") <= 1000000)  # Filter data entry errors
    )


def clean_and_transform(df):
    """Apply cleaning and transformation rules using function composition"""
    logger.info("Starting data cleaning and transformation")

    # Function composition pipeline
    df_cleaned = (df
                  .transform(clean_invalid_markers)
                  .transform(lambda df: standardize_region_names(df, "kabupaten_kota"))
                  .transform(convert_data_types)
                  .transform(filter_valid_records)
                  .transform(calculate_derived_metrics)
                  .transform(lambda df: clean_text_column(df, "kabupaten_kota"))
                  .transform(lambda df: clean_text_column(df, "komoditas"))
                  .dropDuplicates()
                  .select("kabupaten_kota", "komoditas", "produksi", "luas_panen",
                         "produktivitas", "produktivitas_per_ha", "tahun"))

    logger.info(f"Cleaning completed. Rows after cleaning: {df_cleaned.count()}")
    return df_cleaned


def validate_output_data(df):
    """Validate cleaned data quality"""
    logger.info("Validating cleaned data quality")

    required_columns = list(QUALITY_CONFIG["required_columns"].get("produksi", []))
    validation_passed = validate_data_quality(df, required_columns, min_rows=100)

    if not validation_passed:
        raise ValueError("Output data validation failed")

    # Additional business rule validations
    invalid_productivity = df.filter(
        (col("produktivitas_per_ha") <= 0)
        | (col("produktivitas_per_ha") > 100)  # Extremely high productivity threshold
    ).count()

    if invalid_productivity > 0:
        logger.warning(
            f"Found {invalid_productivity} records with suspicious productivity values"
        )

    logger.info("Output data validation passed")
    return True


def save_to_silver(spark, df):
    """Save cleaned data to silver layer"""
    output_path = f"{str(HDFS_CONFIG['data_paths']['silver'])}/produksi_pangan"
    database = str(HIVE_CONFIG["databases"]["silver"])
    table_name = "produksi_pangan"

    logger.info(f"Saving cleaned data to silver layer: {output_path}")

    # Write parquet files partitioned by year
    write_parquet_to_hdfs(
        df,
        output_path,
        partition_by=str(FILE_CONFIG["partition_column"]),
        mode=str(FILE_CONFIG["write_mode"]),
    )

    # Define table schema
    columns = {
        "kabupaten_kota": "STRING",
        "komoditas": "STRING",
        "produksi": "DOUBLE",
        "luas_panen": "DOUBLE",
        "produktivitas": "DOUBLE",
        "produktivitas_per_ha": "DOUBLE",
    }

    partition_columns = {"tahun": "INT"}

    # Create Hive table
    create_hive_table(
        spark,
        database,
        table_name,
        columns,
        partition_columns,
        output_path,
        "PARQUET",
    )

    logger.info(f"Successfully created Hive table: {database}.{table_name}")


def run_pipeline():
    """Execute the complete cleaning pipeline using functional composition"""
    spark = None
    try:
        logger.info("Starting production data cleaning pipeline")

        # Initialize Spark session
        spark = SparkSessionManager.get_session("CleanProduksiData")

        # Execute pipeline
        df_bronze = load_bronze_data(spark)
        validate_input_data(df_bronze)
        df_silver = clean_and_transform(df_bronze)
        validate_output_data(df_silver)
        save_to_silver(spark, df_silver)

        logger.info("Production data cleaning pipeline completed successfully")

        # Print summary statistics
        input_count = df_bronze.count()
        output_count = df_silver.count()
        reduction_pct = (input_count - output_count) / input_count * 100

        logger.info("Pipeline Summary:")
        logger.info(f"- Input rows: {input_count}")
        logger.info(f"- Output rows: {output_count}")
        logger.info(f"- Data reduction: {reduction_pct:.2f}%")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        if spark:
            SparkSessionManager.stop_session()


def main():
    """Main entry point"""
    run_pipeline()


if __name__ == "__main__":
    main()
