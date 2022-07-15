"""
This is a boilerplate pipeline
generated using Kedro 0.18.2
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pyspark.pandas as ps

log = logging.getLogger(__name__)

from audit_report_utils import _restructure_tables_parameters, _arrange_distribution_columns


def count_missing_data(
    staged_dataset: ps.DataFrame, table_parameter: Dict
) -> pd.DataFrame:
    """Maps missing data that cannot be automatically flagged as missing to "NA" and then calculate the count and
    percentage of missing data for each column

    Args:
        staged_dataset: Pandas on Spark Dataframe to count missing data column-wise
        table_parameter: Dictionary of parameters for the staged table, which includes "null_values"
        i.e. values (e.g. "Missing", "Blank", "-") should be mapped as missing

    Returns: Pandas DataFrame contains MISSING_COUNT and MISSING_PERCENTAGE
    """
    # missing value remapping
    for column_name in staged_dataset.columns:
        null_values_mapping = table_parameter.get(
            column_name, table_parameter["_default_"]
        ).get("null_values", [])
        if len(null_values_mapping) > 0:
            staged_dataset[column_name] = staged_dataset[column_name].replace(
                null_values_mapping, None
            )

    # calculating the null value counts and percentage
    missing_value_count = staged_dataset.isnull().sum().to_pandas()
    missing_value_percent = (
        100 * staged_dataset.isnull().sum() / len(staged_dataset)
    ).to_pandas()
    missing_report = pd.concat([missing_value_count, missing_value_percent], axis=1)
    missing_report.columns = ["MISSING_COUNT", "MISSING_PERCENTAGE"]
    missing_report.index = missing_report.index.set_names("VARIABLE")
    missing_report.reset_index(inplace=True)

    return missing_report


def numeric_distribution(
    staged_dataset: ps.DataFrame, table_parameter: Dict
) -> pd.DataFrame:
    """Find the distribution for each of the numeric columns

    Args:
        staged_dataset: Pandas on Spark Dataframe to calculate distribution for the numeric columns
        table_parameter: Dictionary of parameters for the staged table, which includes "percentiles" to be calculated

    Returns: Pandas DataFrame contains count, mean, min, max and percentiles
    """
    # Using describe() to calculate the distribution for numeric values

    numerical_distribution_stacked = pd.DataFrame()
    if len(staged_dataset.columns) > 0:
        for column_name in staged_dataset.columns:
            distribution_percentiles = table_parameter.get(
                column_name, table_parameter["_default_"]
            ).get("percentiles", [])
            numerical_distribution = (
                staged_dataset[column_name]
                .describe(percentiles=distribution_percentiles)
                .to_frame()
                .T.to_pandas()
            )

            numerical_distribution.index = numerical_distribution.index.set_names(
                "VARIABLE"
            )
            numerical_distribution.reset_index(inplace=True)
            numerical_distribution.columns = map(
                str.upper, numerical_distribution.columns
            )

            numerical_distribution_stacked = pd.concat(
                [numerical_distribution_stacked, numerical_distribution],
                ignore_index=True,
            )

    # rename the std to standard_deviation in column
    numerical_distribution_stacked.rename(
        columns={"STD": "STANDARD_DEVIATION"}, inplace=True
    )
    return numerical_distribution_stacked


def categorical_value_counts(
    staged_dataset: ps.DataFrame, table_parameter: Dict
) -> pd.DataFrame:
    """Find the value counts for each of the categorical columns

    Args:
        staged_dataset: Pandas on Spark Dataframe to calculate value counts for the categorical columns
        table_parameter: Dictionary of parameters for the staged table, which includes
            1) "value_counts_drop_na": True or False indicating whether to include NA as one of the level while
            counting the value counts
            2) "value_counts_display": Integer representing the max levels of value counts to be shown for each column

    Return: Pandas DataFrame contains LEVEL and LEVEL_FREQUENCY
    """
    value_counts_stacked = pd.DataFrame()
    if len(staged_dataset.columns) > 0:
        for column_name in staged_dataset.columns:
            value_counts_drop_na = table_parameter.get(
                column_name, table_parameter["_default_"]
            ).get("value_counts_drop_na", [])
            value_counts_display_cutoff = table_parameter.get(
                column_name, table_parameter["_default_"]
            ).get("value_counts_display_cutoff", [])

            value_counts = (
                staged_dataset[column_name]
                .value_counts(dropna=value_counts_drop_na, normalize=True)
                .sort_index()
                .to_frame()
            )
            # format the value counts dataframe with three columns: VARIABLE, LEVEL, LEVEL_FREQUENCY
            value_counts = value_counts.to_pandas()
            value_counts.index = value_counts.index.set_names(["LEVEL"])
            value_counts.reset_index(inplace=True)
            value_counts["VARIABLE"] = column_name
            value_counts = value_counts.rename(columns={column_name: "LEVEL_FREQUENCY"})

            # trim down if more than value_counts_display levels
            if len(value_counts) > value_counts_display_cutoff:
                last_nrows = len(value_counts) - value_counts_display_cutoff
                sum_freq = value_counts.tail(last_nrows)["LEVEL_FREQUENCY"].sum()
                value_counts = value_counts.head(value_counts_display_cutoff).append(
                    {
                        "VARIABLE": column_name,
                        "LEVEL": f"LAST_{last_nrows}_LEVELS",
                        "LEVEL_FREQUENCY": sum_freq,
                    },
                    ignore_index=True,
                )

            value_counts_stacked = pd.concat(
                [value_counts_stacked, value_counts], ignore_index=True
            )

    return value_counts_stacked


def compute_mode_unique(staged_dataset: ps.DataFrame) -> pd.DataFrame:
    """Find the mode and number of distinct values for each column

    Args:
        staged_dataset: Pandas on Spark Dataframe to calculate mode and number of distinct values

    Return: Pandas DataFrame contains NO_DISTINCT_CATEGORIES and MODE
    """
    frequency_report = pd.DataFrame(
        columns=["VARIABLE", "NO_DISTINCT_CATEGORIES", "MODE"]
    )
    for col in staged_dataset:
        # compute mode of the column
        modes = staged_dataset[col].mode()
        flattened_modes = modes.to_list()

        # For certain ID or timestamp values, the mode is all unique values listed. We don't want this in the Audit
        # Report. If there are more than 3 modes, we direct the user to the value counts report instead - it will be
        # easier ot identify/observe multi-modal situations through that format
        if len(flattened_modes) > 3:
            flattened_modes = "More than 3 modes identified - Please refer to Value Counts Audit Report"

        # compute number of distinct categories of the column
        unique_category_count = staged_dataset[col].nunique()

        # append to generate the frequency report
        frequency_report = frequency_report.append(
            {
                "VARIABLE": col,
                "NO_DISTINCT_CATEGORIES": unique_category_count,
                "MODE": flattened_modes,
            },
            ignore_index=True,
        )

    return frequency_report


def generate_audit_report(
    audit_report_input_datasets: Dict, summary_statistics_param: Dict
) -> Dict[str, pd.DataFrame]:
    """Iterate through all staged datasets inputs to generate comprehensive audit reports.
    We are going to generate two audit reports: one called audit_report_distribution which includes the summary
    statistics such as mean, mode, range, and percentile distribution for all variables in the staged tables;
    the other one called audit_value_counts shows the frequency of distinct levels for ONLY the categorical variables

    Args:
        audit_report_input_datasets: Dictionary of Pandas on Spark Staging datasets to generate audit reports
        summary_statistics_param: Parameters specified under in audit_report_parameters.yml

    Return: Two Pandas DataFrame one for distribution and the other one for value counts
    """

    audit_report_distribution_stacked = pd.DataFrame()
    audit_value_counts_stacked = pd.DataFrame()
    log.info("BEGIN CREATING AUDIT REPORT...")
    for staged_table_name, staged_table in sorted(audit_report_input_datasets.items()):
        log.info("Processing staged table -> {}".format(staged_table_name))

        # Restructure the parameters
        staged_table_parameter = _restructure_tables_parameters(
            audit_report_parameters=summary_statistics_param,
            table_name=staged_table_name,
        )

        # Count missing data
        log.info("METRIC - Calculating missing data")
        audit_missing_df = count_missing_data(
            staged_dataset=staged_table, table_parameter=staged_table_parameter
        )

        # Calculate distribution for numerical variables
        log.info("METRIC - Calculating numeric distributions (e.g. percentiles)")
        numeric_vars_dataframe = staged_table.select_dtypes(include=["int", "float"])
        audit_distribution_df = numeric_distribution(
            staged_dataset=numeric_vars_dataframe,
            table_parameter=staged_table_parameter,
        )

        # Calculate value counts for categorical variables
        log.info("METRIC - Calculating categorical variable relative counts")
        categorical_vars_dataframe = staged_table.select_dtypes(
            exclude=["int", "float"]
        )
        audit_value_counts = categorical_value_counts(
            staged_dataset=categorical_vars_dataframe,
            table_parameter=staged_table_parameter,
        )

        # Find the distinct values and mode
        log.info("METRIC - Calculating mode (OR top 3 most frequent value")
        audit_frequency_df = compute_mode_unique(staged_dataset=staged_table)

        # merge all summary statistics into one dataframe
        audit_report_distribution = audit_missing_df.merge(
            audit_frequency_df, how="outer", on="VARIABLE"
        )
        # Added to avoid merging empty dataframes for staged tables without numeric variables
        if len(audit_distribution_df) > 0:
            audit_report_distribution = audit_report_distribution.merge(
                audit_distribution_df, how="outer", on="VARIABLE"
            )

        # TODO: How to specify the host and filepath name in parameters, or dynamically read in
        audit_report_distribution["HOST"] = "S3://dummy_bucket"
        audit_report_distribution["FILEPATH"] = "03_stage"
        audit_value_counts["HOST"] = "S3://dummy_bucket"
        audit_value_counts["FILEPATH"] = "03_stage"

        audit_report_distribution["PARTITION_NAME"] = staged_table_name
        audit_report_distribution["TOTAL_NO_OF_OBSERVATIONS"] = len(staged_table)

        audit_value_counts["PARTITION_NAME"] = staged_table_name

        log.info(
            "Successfully combined METRICS for table -> {}".format(staged_table_name)
        )

        # Concatenate results from different tables into one dataframe for each file.
        # The audit report distribution provides broad metrics related to summary statistics, distributions,
        # data missingness and unique values
        audit_report_distribution_stacked = pd.concat(
            [audit_report_distribution_stacked, audit_report_distribution],
            ignore_index=True,
        )
        log.info(
            "Successfully stacked distribution values from table -> {}".format(
                staged_table_name
            )
        )

        # The audit value counts report helps provide the relative frequencies of unique categories in non-numeric
        # values.
        audit_value_counts_stacked = pd.concat(
            [audit_value_counts_stacked, audit_value_counts], ignore_index=True
        )
        log.info(
            "Successfully stacked value counts from table -> {}".format(
                staged_table_name
            )
        )

    # reshuffle the orders of columns and control what to be displayed vs not
    log.info("RE-ARRANGING AUDIT REPORT COLUMN ORDER")
    (
        audit_report_distribution_output,
        audit_value_counts_output,
    ) = _arrange_distribution_columns(
        output_option_parameters=summary_statistics_param["output_option"],
        audit_report_distribution=audit_report_distribution_stacked,
        audit_value_counts=audit_value_counts_stacked,
    )

    # The distribution report helps inform / define how missing data or outlier are treated. We add these fields
    # for the Data Scientist to define the strategies to treat missing data / outliers
    # We decided to hard code these to avoid custom namings in a parameters file that might break subsequent automated ]
    # steps
    audit_report_input_fields = [
        "missing_drop_column_flag",
        "missing_impute_method",
        "missing_impute_values",
        "outlier_left_bound_value",
        "outlier_left_bound_method",
        "outlier_right_bound_value",
        "outlier_right_bound_method",
    ]

    audit_report_distribution_output[audit_report_input_fields] = ""
    log.info("SUCCESSFULLY GENERATED AUDIT REPORT FILES to be stored in 98_report/")

    multi_excel_report = {
        "distribution_report": audit_report_distribution_output,
        "value_counts_report": audit_value_counts_output,
    }

    return multi_excel_report
