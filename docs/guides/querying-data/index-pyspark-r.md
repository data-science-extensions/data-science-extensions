# Data Querying for Data Scientists


### A Comprehensive Guide of using Pandas, SQL, PySpark, and Polars for Data Manipulation Techniques, with Practical Examples and Visualisations

If you wanted to run the code yourself, you can download just that Jupyter notebook:

[![](./../../assets/icons/dse.svg){width=70px}<br>Download<br>- ALL](./index-r.ipynb){ :download .md-button .md-button-fifth }
[![](./../../assets/icons/pandas.svg){width=100%}<br>Download<br>- Pandas](./index-pandas-r.ipynb){ :download .md-button .md-button-fifth }
[![](./../../assets/icons/sql.svg){width=100%}<br>Download<br>- SQL](./index-sql-r.ipynb){ :download .md-button .md-button-fifth }
[![](./../../assets/icons/spark.svg){width=100%}<br>Download<br>- PySpark](./index-pyspark-r.ipynb){ :download .md-button .md-button-fifth }
[![](./../../assets/icons/polars.svg){width=100%}<br>Download<br>- Polars](./index-polars-r.ipynb){ :download .md-button .md-button-fifth }

Or you can follow along on this page...


## Introduction

Working as a Data Scientist or Data Engineer often involves querying data from various sources. There are many tools and libraries available to perform these tasks, each with its own strengths and weaknesses. Also, there are many different ways to achieve similar results, depending on the tool or library used. It's important to be familiar with these different methods to choose the best one for your specific use case.

This article provides a comprehensive guide on how to query data using different tools and libraries, including Pandas, SQL, PySpark, and Polars. Each section will cover the setup, data creation, and various querying techniques such as filtering, grouping, joining, window functions, ranking, and sorting. The output will be identical across all tools, but the transformations will be implemented using the specific syntax and features of each library. Therefore allowing you to compare the different approaches and understand the nuances of each method.


## Overview of the Different Libraries

Before we dive into the querying techniques, let's take a moment to understand the different libraries and tools we will be using in this article. Each library has its own strengths and weaknesses, and understanding these can help you choose the right tool for your specific use case.

Throughout this article, you can easily switch between the different libraries by selecting the appropriate tab. Each section will provide the same functionality, but implemented using the specific syntax and features of each library.

### PySpark

[PySpark] is the Python API for Apache Spark, a distributed computing framework that allows for large-scale data processing. PySpark provides a high-level interface for working with Spark, making it easier to write distributed data processing applications in Python. It is particularly well-suited for big data processing and analytics.

PySpark provides a DataFrame API similar to Pandas, but it is designed to work with large datasets that do not fit into memory. It allows for distributed data processing across a cluster of machines, making it suitable for big data applications. PySpark supports various data sources, including [HDFS], [S3], [ADLS], and [JDBC], and provides powerful features for filtering, grouping, joining, and aggregating data.

While PySpark is a powerful tool for big data processing, it can be more complex to set up and use compared to Pandas. It requires a Spark cluster and may have a steeper learning curve for those unfamiliar with distributed computing concepts. However, it is an excellent choice for processing large datasets and performing complex data transformations.

## Setup

Before we start querying data, we need to set up our environment. This includes importing the necessary libraries, creating sample data, and defining constants that will be used throughout the article. The following sections will guide you through this setup process. The code for this article is also available on GitHub: [querying-data][querying-data].

### PySpark

```python {.pyspark linenums="1" title="Setup"}
# StdLib Imports
from typing import Any

# Third Party Imports
import numpy as np
from plotly import express as px, graph_objects as go, io as pio
from pyspark.sql import (
    DataFrame as psDataFrame,
    SparkSession,
    Window,
    functions as F,
    types as T,
)


# Set seed for reproducibility
np.random.seed(42)

# Determine the number of records to generate
n_records = 100

# Set default Plotly template
pio.templates.default = "simple_white+gridon"
```

Once the setup is complete, we can proceed to create our sample data. This data will be used for querying and will be consistent across all libraries. All tables will be created from scratch with randomly generated data to simulate a real-world scenario. This is to ensure that the examples are self-contained and can be run without any external dependencies, and also there is no issues about data privacy or security.

For the below data creation steps, we will be defining the tables using Python dictionaries. Each dictionary will represent a table, with keys as column names and values as lists of data. We will then convert these dictionaries into DataFrames or equivalent structures in each library.

First, we will create a sales fact table. This table will contain information about sales transactions, including the date, customer ID, product ID, category, sales amount, and quantity sold.

```python {.python linenums="1" title="Create Sales Fact Data"}
sales_data: dict[str, Any] = {
    "date": pd.date_range(start="2023-01-01", periods=n_records, freq="D"),
    "customer_id": np.random.randint(1, 100, n_records),
    "product_id": np.random.randint(1, 50, n_records),
    "category": np.random.choice(["Electronics", "Clothing", "Food", "Books", "Home"], n_records),
    "sales_amount": np.random.uniform(10, 1000, n_records).round(2),
    "quantity": np.random.randint(1, 10, n_records),
}
```

Next, we will create a product dimension table. This table will contain information about products, including the product ID, name, price, category, and supplier ID.

```python {.python linenums="1" title="Create Product Dimension Data"}
product_data: dict[str, Any] = {
    "product_id": np.arange(1, 51),
    "product_name": [f"Product {i}" for i in range(1, 51)],
    "price": np.random.uniform(10, 500, 50).round(2),
    "category": np.random.choice(["Electronics", "Clothing", "Food", "Books", "Home"], 50),
    "supplier_id": np.random.randint(1, 10, 50),
}
```

Finally, we will create a customer dimension table. This table will contain information about customers, including the customer ID, name, city, state, and segment.

```python {.python linenums="1" title="Create Customer Dimension Data"}
customer_data: dict[str, Any] = {
    "customer_id": np.arange(1, 101),
    "customer_name": [f"Customer {i}" for i in range(1, 101)],
    "city": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], 100),
    "state": np.random.choice(["NY", "CA", "IL", "TX", "AZ"], 100),
    "segment": np.random.choice(["Consumer", "Corporate", "Home Office"], 100),
}
```

Now that we have our sample data created, we can proceed to the querying section. Each of the following sections will demonstrate how to perform similar operations using the different libraries and methods, allowing you to compare and contrast their capabilities.


## Create the DataFrames

### PySpark

Spark DataFrames are similar to Pandas DataFrames, but they are designed to work with large datasets that do not fit into memory. They can be distributed across a cluster of machines, allowing for parallel processing of data.

To create the dataframes in PySpark, we will use the data we generated earlier. We will first create a Spark session, which is the entry point to using PySpark. Then, we will parse the dictionaries into PySpark DataFrames, which will allow us to perform various data manipulation tasks.

The PySpark session is created using the [`.builder`][pyspark-builder] method on the [`SparkSession`][pyspark-sparksession] class, which allows us to configure the session with various options such as the application name. The [`.getOrCreate()`][pyspark-getorcreate] method is used to either get an existing session or create a new one if it doesn't exist.

```python {.pyspark linenums="1" title="Create Spark Session"}
spark: SparkSession = SparkSession.builder.appName("SalesAnalysis").getOrCreate()
```

Once the Spark session is created, we can create the DataFrames from the dictionaries. We will  use the [`.createDataFrame()`][pyspark-createdataframe] method on the Spark session to convert the dictionaries into PySpark DataFrames. The [`.createDataFrame()`][pyspark-createdataframe] method is expecting the data to be oriented by _row_. Meaning that the data should be in the form of a list of dictionaries, where each dictionary represents a row of data. However, we currently have our data is oriented by _column_, where the dictionarieshave keys as column names and values as lists of data. Therefore, we will first need to convert the dictionaries from _column_ orientation to _row_ orientation. The easiest way to do this is by parse'ing the data to a Pandas DataFrames, and then using that to create our PySpark DataFrames from there.

A good description of how to create PySpark DataFrames from Python Dictionaries can be found in the PySpark documentation: [PySpark Create DataFrame From Dictionary][pyspark-create-dataframe-from-dict].

```python {.pyspark linenums="1" title="Create DataFrames"}
df_sales_ps: psDataFrame = spark.createDataFrame(pd.DataFrame(sales_data))
df_product_ps: psDataFrame = spark.createDataFrame(pd.DataFrame(product_data))
df_customer_ps: psDataFrame = spark.createDataFrame(pd.DataFrame(customer_data))
```

Once the data is created, we can check that it has been loaded correctly by displaying the first few rows of each DataFrame. To do this, we will use the [`.show()`][pyspark-show] method to display the first `5` rows of each DataFrame. The [`.show()`][pyspark-show] method is used to display the data in a tabular format, similar to how it would be displayed in a SQL database.

```python {.pyspark linenums="1" title="Check Sales DataFrame"}
print(f"Sales DataFrame: {df_sales_ps.count()}")
df_sales_ps.show(5)
print(df_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales DataFrame: 100
```

```txt
+-------------------+-----------+----------+-----------+------------+--------+
|               date|customer_id|product_id|   category|sales_amount|quantity|
+-------------------+-----------+----------+-----------+------------+--------+
|2023-01-01 00:00:00|         52|        45|       Food|      490.76|       7|
|2023-01-02 00:00:00|         93|        41|Electronics|      453.94|       5|
|2023-01-03 00:00:00|         15|        29|       Home|      994.51|       5|
|2023-01-04 00:00:00|         72|        15|Electronics|      184.17|       7|
|2023-01-05 00:00:00|         61|        45|       Food|       27.89|       9|
+-------------------+-----------+----------+-----------+------------+--------+
only showing top 10 rows
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-01 00:00:00 |          52 |         45 | Food        |       490.76 |        7 |
|    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
|    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
|    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
|    4 | 2023-01-05 00:00:00 |          61 |         45 | Food        |        27.89 |        9 |

</div>

```python {.pyspark linenums="1" title="Check Product DataFrame"}
print(f"Product DataFrame: {df_product_ps.count()}")
df_product_ps.show(5)
print(df_product_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Product DataFrame: 50
```

```txt
+----------+------------+------+--------+-----------+
|product_id|product_name| price|category|supplier_id|
+----------+------------+------+--------+-----------+
|         1|   Product 1|257.57|    Food|          8|
|         2|   Product 2|414.96|Clothing|          5|
|         3|   Product 3|166.82|Clothing|          8|
|         4|   Product 4|448.81|    Food|          4|
|         5|   Product 5|200.71|    Food|          8|
+----------+------------+------+--------+-----------+
only showing top 5 rows
```

|      | product_id | product_name |  price | category | supplier_id |
| ---: | ---------: | :----------- | -----: | :------- | ----------: |
|    0 |          1 | Product 1    | 257.57 | Food     |           8 |
|    1 |          2 | Product 2    | 414.96 | Clothing |           5 |
|    2 |          3 | Product 3    | 166.82 | Clothing |           8 |
|    3 |          4 | Product 4    | 448.81 | Food     |           4 |
|    4 |          5 | Product 5    | 200.71 | Food     |           8 |

</div>

```python {.pyspark linenums="1" title="Check Customer DataFrame"}
print(f"Customer DataFrame: {df_customer_ps.count()}")
df_customer_ps.show(5)
print(df_customer_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Customer DataFrame: 100
```

```txt
+-----------+-------------+-----------+-----+-----------+
|customer_id|customer_name|       city|state|    segment|
+-----------+-------------+-----------+-----+-----------+
|          1|   Customer 1|    Phoenix|   NY|  Corporate|
|          2|   Customer 2|    Phoenix|   CA|Home Office|
|          3|   Customer 3|    Phoenix|   NY|Home Office|
|          4|   Customer 4|Los Angeles|   NY|   Consumer|
|          5|   Customer 5|Los Angeles|   IL|Home Office|
+-----------+-------------+-----------+-----+-----------+
only showing top 5 rows
```

|      | customer_id | customer_name | city        | state | segment     |
| ---: | ----------: | :------------ | :---------- | :---- | :---------- |
|    0 |           1 | Customer 1    | Phoenix     | NY    | Corporate   |
|    1 |           2 | Customer 2    | Phoenix     | CA    | Home Office |
|    2 |           3 | Customer 3    | Phoenix     | NY    | Home Office |
|    3 |           4 | Customer 4    | Los Angeles | NY    | Consumer    |
|    4 |           5 | Customer 5    | Los Angeles | IL    | Home Office |

</div>

## 1. Filtering and Selecting

This first section will demonstrate how to filter and select data from the DataFrames. This is a common operation in data analysis, allowing us to focus on specific subsets of the data.

### PySpark

In PySpark, we can use the [`.filter()`][pyspark-filter] (or the [`.where()`][pyspark-where]) method to filter rows based on specific conditions. This process is effectively doing a boolean indexing operation to filter the DataFrame. The syntax is similar to SQL, where we can specify the condition as a string or using column expressions. In the below example, we filter for sales in the "Electronics" category.

For more information about filtering in PySpark, see the [PySpark documentation on filtering][pyspark-filtering].

```python {.pyspark linenums="1" title="Filter sales for a specific category"}
electronics_sales_ps: psDataFrame = df_sales_ps.filter(df_sales_ps["category"] == "Electronics")
print(f"Number of Electronics Sales: {electronics_sales_ps.count()}")
electronics_sales_ps.show(5)
print(electronics_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Number of Electronics Sales: 28
```

```txt
+-------------------+-----------+----------+-----------+------------+--------+
|               date|customer_id|product_id|   category|sales_amount|quantity|
+-------------------+-----------+----------+-----------+------------+--------+
|2023-01-02 00:00:00|         93|        41|Electronics|      453.94|       5|
|2023-01-04 00:00:00|         72|        15|Electronics|      184.17|       7|
|2023-01-09 00:00:00|         75|         9|Electronics|      746.73|       2|
|2023-01-11 00:00:00|         88|         1|Electronics|      314.98|       9|
|2023-01-12 00:00:00|         24|        44|Electronics|      547.11|       8|
+-------------------+-----------+----------+-----------+------------+--------+
only showing top 5 rows
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
|    1 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
|    2 | 2023-01-09 00:00:00 |          75 |          9 | Electronics |       746.73 |        2 |
|    3 | 2023-01-11 00:00:00 |          88 |          1 | Electronics |       314.98 |        9 |
|    4 | 2023-01-12 00:00:00 |          24 |         44 | Electronics |       547.11 |        8 |

</div>

We can also use numerical filtering, as you can see in the next example, where we filter for sales amounts greater than $500.

### PySpark

When it comes to numerical filtering in PySpark, the process is similar to the previous example, where we use the [`.filter()`][pyspark-filter] (or [`.where()`][pyspark-where]) method to filter rows based on a given condition, but here we use a numerical value instead of a string value. In the below example, we filter for sales amounts greater than `500`.

Also note here that we have parsed a string value to the [`.filter()`][pyspark-filter] method, instead of using the pure-Python syntax as shown above. This is because the [`.filter()`][pyspark-filter] method can accept a SQL-like string expression. This is a common practice in PySpark to parse a SQL-like string to a PySpark method.

```python {.pyspark linenums="1" title="Filter for high value transactions"}
high_value_sales_ps: psDataFrame = df_sales_ps.filter("sales_amount > 500")
print(f"Number of high-value Sales: {high_value_sales_ps.count()}")
high_value_sales_ps.show(5)
print(high_value_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Number of high-value Sales: 43
```

```txt
+-------------------+-----------+----------+-----------+------------+--------+
|               date|customer_id|product_id|   category|sales_amount|quantity|
+-------------------+-----------+----------+-----------+------------+--------+
|2023-01-03 00:00:00|         15|        29|       Home|      994.51|       5|
|2023-01-09 00:00:00|         75|         9|Electronics|      746.73|       2|
|2023-01-10 00:00:00|         75|        24|      Books|      723.73|       6|
|2023-01-12 00:00:00|         24|        44|Electronics|      547.11|       8|
|2023-01-13 00:00:00|          3|         8|   Clothing|      513.73|       5|
+-------------------+-----------+----------+-----------+------------+--------+
only showing top 5 rows
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
|    1 | 2023-01-09 00:00:00 |          75 |          9 | Electronics |       746.73 |        2 |
|    2 | 2023-01-10 00:00:00 |          75 |         24 | Books       |       723.73 |        6 |
|    3 | 2023-01-12 00:00:00 |          24 |         44 | Electronics |       547.11 |        8 |
|    4 | 2023-01-13 00:00:00 |           3 |          8 | Clothing    |       513.73 |        5 |

</div>

In addition to subsetting a table by rows (aka _filtering_), we can also subset a table by columns (aka _selecting_). This allows us to create a new DataFrame with only the relevant columns we want to work with. This is useful when we want to focus on specific attributes of the data, such as dates, categories, or sales amounts.


### PySpark

To select specific columns in PySpark, we can use the [`.select()`][pyspark-select] method to specify the columns we want to keep in the DataFrame. This allows us to create a new DataFrame with only the relevant columns. The syntax is similar to SQL, where we can specify the column names as strings.

```python {.pyspark linenums="1" title="Select specific columns"}
sales_summary_ps: psDataFrame = df_sales_ps.select("date", "category", "sales_amount")
print(f"Sales Summary DataFrame: {sales_summary_ps.count()}")
sales_summary_ps.show(5)
print(sales_summary_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales Summary DataFrame: 100
```

```txt
+-------------------+-----------+------------+
|               date|   category|sales_amount|
+-------------------+-----------+------------+
|2023-01-01 00:00:00|       Food|      490.76|
|2023-01-02 00:00:00|Electronics|      453.94|
|2023-01-03 00:00:00|       Home|      994.51|
|2023-01-04 00:00:00|Electronics|      184.17|
|2023-01-05 00:00:00|       Food|       27.89|
+-------------------+-----------+------------+
only showing top 5 rows
```

|      | date                | category    | sales_amount |
| ---: | :------------------ | :---------- | -----------: |
|    0 | 2023-01-01 00:00:00 | Food        |       490.76 |
|    1 | 2023-01-02 00:00:00 | Electronics |       453.94 |
|    2 | 2023-01-03 00:00:00 | Home        |       994.51 |
|    3 | 2023-01-04 00:00:00 | Electronics |       184.17 |
|    4 | 2023-01-05 00:00:00 | Food        |        27.89 |

</div>

## 2. Grouping and Aggregation

The second section will cover grouping and aggregation techniques. These operations are essential for summarizing data and extracting insights from large datasets.

### PySpark

In PySpark, we can use the [`.agg()`][pyspark-agg] method to perform aggregation operations on DataFrames. This method allows us to apply multiple aggregation functions to different columns in a single operation.

```python {.pyspark linenums="1" title="Basic aggregation"}
sales_stats_ps: psDataFrame = df_sales_ps.agg(
    F.sum("sales_amount").alias("sales_sum"),
    F.avg("sales_amount").alias("sales_mean"),
    F.expr("MIN(sales_amount) AS sales_min"),
    F.expr("MAX(sales_amount) AS sales_max"),
    F.count("*").alias("sales_count"),
    F.expr("SUM(quantity) AS quantity_sum"),
    F.expr("AVG(quantity) AS quantity_mean"),
    F.min("quantity").alias("quantity_min"),
    F.max("quantity").alias("quantity_max"),
)
print(f"Sales Statistics: {sales_stats_ps.count()}")
sales_stats_ps.show(5)
print(sales_stats_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales Statistics: 1
```

```txt
+---------+----------+---------+---------+-----------+------------+-------------+------------+------------+
|sales_sum|sales_mean|sales_min|sales_max|sales_count|quantity_sum|quantity_mean|quantity_min|quantity_max|
+---------+----------+---------+---------+-----------+------------+-------------+------------+------------+
| 48227.05|  482.2705|    15.13|   994.61|        100|         464|         4.64|           1|           9|
+---------+----------+---------+---------+-----------+------------+-------------+------------+------------+
```

|      | sales_sum | sales_mean | sales_min | sales_max | sales_count | quantity_sum | quantity_mean | quantity_min | quantity_max |
| ---: | --------: | ---------: | --------: | --------: | ----------: | -----------: | ------------: | -----------: | -----------: |
|    0 |   48227.1 |    482.271 |     15.13 |    994.61 |         100 |          464 |          4.64 |            1 |            9 |

</div>

It is also possible to group the data by a specific column and then apply aggregation functions to summarize the data by group.

### PySpark

In PySpark, we can use the [`.groupBy()`][pyspark-groupby] method to group data by one or more columns and then apply aggregation functions using the [`.agg()`][pyspark-groupby-agg] method.

```python {.pyspark linenums="1" title="Group by category and aggregate"}
category_sales_ps: psDataFrame = df_sales_ps.groupBy("category").agg(
    F.sum("sales_amount").alias("total_sales"),
    F.avg("sales_amount").alias("average_sales"),
    F.count("*").alias("transaction_count"),
    F.sum("quantity").alias("total_quantity"),
)
print(f"Category Sales Summary: {category_sales_ps.count()}")
category_sales_ps.show(5)
print(category_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Category Sales Summary: 5
```

```txt
+-----------+------------------+------------------+-----------------+--------------+
|   category|       total_sales|     average_sales|transaction_count|total_quantity|
+-----------+------------------+------------------+-----------------+--------------+
|       Home| 6343.889999999999| 704.8766666666666|                9|            40|
|       Food|          12995.57| 541.4820833333333|               24|           115|
|Electronics|11407.449999999999|407.40892857142853|               28|           147|
|   Clothing|7325.3099999999995|457.83187499999997|               16|            62|
|      Books|          10154.83|  441.514347826087|               23|           100|
+-----------+------------------+------------------+-----------------+--------------+
```

|      | category    | total_sales | average_sales | transaction_count | total_quantity |
| ---: | :---------- | ----------: | ------------: | ----------------: | -------------: |
|    0 | Home        |     6343.89 |       704.877 |                 9 |             40 |
|    1 | Food        |     12995.6 |       541.482 |                24 |            115 |
|    2 | Electronics |     11407.4 |       407.409 |                28 |            147 |
|    3 | Clothing    |     7325.31 |       457.832 |                16 |             62 |
|    4 | Books       |     10154.8 |       441.514 |                23 |            100 |

</div>

We can rename the columns for clarity by simply assigning new names.

### PySpark

In PySpark, we can use the [`.withColumnsRenamed()`][pyspark-withcolumnsrenamed] method to rename columns in a DataFrame. This allows us to provide more descriptive names for the aggregated columns.

```python {.pyspark linenums="1" title="Rename columns for clarity"}
category_sales_renamed_ps: psDataFrame = category_sales_ps.withColumnsRenamed(
    {
        "total_sales": "Total Sales",
        "average_sales": "Average Sales",
        "transaction_count": "Transaction Count",
        "total_quantity": "Total Quantity",
    }
)
print(f"Renamed Category Sales Summary: {category_sales_renamed_ps.count()}")
category_sales_renamed_ps.show(5)
print(category_sales_renamed_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Renamed Category Sales Summary: 5
```

```txt
+-----------+------------------+------------------+-----------------+--------------+
|   category|       Total Sales|     Average Sales|Transaction Count|Total Quantity|
+-----------+------------------+------------------+-----------------+--------------+
|       Home| 6343.889999999999| 704.8766666666666|                9|            40|
|       Food|          12995.57| 541.4820833333333|               24|           115|
|Electronics|11407.449999999999|407.40892857142853|               28|           147|
|   Clothing|7325.3099999999995|457.83187499999997|               16|            62|
|      Books|          10154.83|  441.514347826087|               23|           100|
+-----------+------------------+------------------+-----------------+--------------+
```

|      | category    | Total Sales | Average Sales | Transaction Count | Total Quantity |
| ---: | :---------- | ----------: | ------------: | ----------------: | -------------: |
|    0 | Home        |     6343.89 |       704.877 |                 9 |             40 |
|    1 | Food        |     12995.6 |       541.482 |                24 |            115 |
|    2 | Electronics |     11407.4 |       407.409 |                28 |            147 |
|    3 | Clothing    |     7325.31 |       457.832 |                16 |             62 |
|    4 | Books       |     10154.8 |       441.514 |                23 |            100 |

</div>

Having aggregated the data, we can now visualize the results using [Plotly][plotly]. This allows us to create interactive visualizations that can help us better understand the data. The simplest way to do this is to use the [Plotly Express][plotly-express] module, which provides a high-level interface for creating visualizations. Here, we have utilised the [`px.bar()`][plotly-bar] function to create a bar chart of the total sales by category.

### PySpark

Plotly is unfortunately not able to directly receive a PySpark DataFrame, so we need to convert it to a Pandas DataFrame first. This is done using the [`.toPandas()`][pyspark-topandas] method, which converts the PySpark DataFrame to a Pandas DataFrame.

```python {.pyspark linenums="1" title="Plot the results"}
fig: go.Figure = px.bar(
    data_frame=category_sales_renamed_ps.toPandas(),
    x="category",
    y="Total Sales",
    title="Total Sales by Category",
    text="Transaction Count",
    labels={"Total Sales": "Total Sales ($)", "category": "Product Category"},
)
fig.write_html("images/pt2_total_sales_by_category_ps.html", include_plotlyjs="cdn", full_html=True)
fig.show()
```

<div class="result" markdown>


</div>



## 3. Joining

The third section will demonstrate how to join DataFrames to combine data from different sources. This is a common operation in data analysis, allowing us to enrich our data with additional information.

Here, we will join the `sales` DataFrame with the `product` DataFrame to get additional information about the products sold.

### PySpark

In PySpark, we can use the [`.join()`][pyspark-join] method to combine rows from two or more DataFrames based on a related column between them. In this case, we will join the `sales` DataFrame with the `product` DataFrame on the `product_id` column.

```python {.pyspark linenums="1" title="Join sales with product data"}
sales_with_product_ps: psDataFrame = df_sales_ps.join(
    other=df_product_ps.select("product_id", "product_name", "price"),
    on="product_id",
    how="left",
)
print(f"Sales with Product Information: {sales_with_product_ps.count()}")
sales_with_product_ps.show(5)
print(sales_with_product_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales with Product Information: 100
```

```txt
+----------+-------------------+-----------+-----------+------------+--------+------------+------+
|product_id|               date|customer_id|   category|sales_amount|quantity|product_name| price|
+----------+-------------------+-----------+-----------+------------+--------+------------+------+
|         1|2023-01-06 00:00:00|         21|   Clothing|      498.95|       5|   Product 1|257.57|
|         1|2023-01-11 00:00:00|         88|Electronics|      314.98|       9|   Product 1|257.57|
|         1|2023-02-11 00:00:00|         55|       Food|       199.0|       5|   Product 1|257.57|
|         1|2023-04-04 00:00:00|         85|       Food|      146.97|       7|   Product 1|257.57|
|         5|2023-01-21 00:00:00|         64|Electronics|      356.58|       5|   Product 5|200.71|
+----------+-------------------+-----------+-----------+------------+--------+------------+------+
only showing top 5 rows
```

|      | product_id | date                | customer_id | category    | sales_amount | quantity | product_name |  price |
| ---: | ---------: | :------------------ | ----------: | :---------- | -----------: | -------: | :----------- | -----: |
|    0 |          1 | 2023-01-11 00:00:00 |          88 | Electronics |       314.98 |        9 | Product 1    | 257.57 |
|    1 |          1 | 2023-02-11 00:00:00 |          55 | Food        |          199 |        5 | Product 1    | 257.57 |
|    2 |          5 | 2023-01-21 00:00:00 |          64 | Electronics |       356.58 |        5 | Product 5    | 200.71 |
|    3 |          5 | 2023-02-18 00:00:00 |          39 | Books       |        79.71 |        8 | Product 5    | 200.71 |
|    4 |          6 | 2023-03-23 00:00:00 |          34 | Electronics |        48.45 |        8 | Product 6    |  15.31 |

</div>

In the next step, we will join the resulting DataFrame with the `customer` DataFrame to get customer information for each sale. This allows us to create a complete view of the sales data, including product and customer details.

### PySpark

This process is similar to the previous step, but now we will extend the `sales_with_product` DataFrame to join it with the `customer` DataFrame on the `customer_id` column. This will give us a complete view of the sales data, including product and customer details.

```python {.pyspark linenums="1" title="Join with customer information to get a complete view"}
complete_sales_ps: psDataFrame = sales_with_product_ps.alias("s").join(
    other=df_customer_ps.select("customer_id", "customer_name", "city", "state").alias("c"),
    on="customer_id",
    how="left",
)
print(f"Complete Sales Data with Customer Information: {complete_sales_ps.count()}")
complete_sales_ps.show(5)
print(complete_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Complete Sales Data with Customer Information: 100
```

```txt
+-----------+----------+-------------------+-----------+------------+--------+------------+------+-------------+-----------+-----+
|customer_id|product_id|               date|   category|sales_amount|quantity|product_name| price|customer_name|       city|state|
+-----------+----------+-------------------+-----------+------------+--------+------------+------+-------------+-----------+-----+
|         39|         5|2023-02-18 00:00:00|      Books|       79.71|       8|   Product 5|200.71|  Customer 39|Los Angeles|   NY|
|         88|         1|2023-01-11 00:00:00|Electronics|      314.98|       9|   Product 1|257.57|  Customer 88|Los Angeles|   TX|
|         85|         1|2023-04-04 00:00:00|       Food|      146.97|       7|   Product 1|257.57|  Customer 85|    Phoenix|   CA|
|         55|         1|2023-02-11 00:00:00|       Food|       199.0|       5|   Product 1|257.57|  Customer 55|Los Angeles|   NY|
|         21|         1|2023-01-06 00:00:00|   Clothing|      498.95|       5|   Product 1|257.57|  Customer 21|Los Angeles|   IL|
+-----------+----------+-------------------+-----------+------------+--------+------------+------+-------------+-----------+-----+
only showing top 5 rows
```

|      | customer_id | product_id | date                | category    | sales_amount | quantity | product_name |  price | customer_name | city        | state |
| ---: | ----------: | ---------: | :------------------ | :---------- | -----------: | -------: | :----------- | -----: | :------------ | :---------- | :---- |
|    0 |          88 |          1 | 2023-01-11 00:00:00 | Electronics |       314.98 |        9 | Product 1    | 257.57 | Customer 88   | Los Angeles | TX    |
|    1 |          55 |          1 | 2023-02-11 00:00:00 | Food        |          199 |        5 | Product 1    | 257.57 | Customer 55   | Los Angeles | NY    |
|    2 |          64 |          5 | 2023-01-21 00:00:00 | Electronics |       356.58 |        5 | Product 5    | 200.71 | Customer 64   | Los Angeles | NY    |
|    3 |          39 |          5 | 2023-02-18 00:00:00 | Books       |        79.71 |        8 | Product 5    | 200.71 | Customer 39   | Los Angeles | NY    |
|    4 |          34 |          6 | 2023-03-23 00:00:00 | Electronics |        48.45 |        8 | Product 6    |  15.31 | Customer 34   | Los Angeles | NY    |

</div>

Once we have the complete sales data, we can calculate the revenue for each sale by multiplying the price and quantity (columns from different tables). We can also compare this calculated revenue with the sales amount to identify any discrepancies.

### PySpark

In PySpark, we can calculate the revenue for each sale by multiplying the `price` and `quantity` columns. We can then compare this calculated revenue with the `sales_amount` column to identify any discrepancies.

Notice here that the syntax for PySpark uses the [`.withColumns`][pyspark-withcolumns] method to add new multiple columns to the DataFrame simultaneously. This method takes a dictionary where the keys are the names of the new columns and the values are the expressions to compute those columns. The methematical computation we have shown here uses two different methods:

1. With the PySpark API, we can use the [`F.col()`][pyspark-col] function to refer to the columns, and multiply them directly
2. With the Spark SQL API, we can use the [`F.expr()`][pyspark-expr] function to write a SQL-like expression for the calculation.

```python {.pyspark linenums="1" title="Calculate revenue and compare with sales amount"}
complete_sales_ps: psDataFrame = complete_sales_ps.withColumns(
    {
        "calculated_revenue": F.col("price") * F.col("quantity"),
        "price_difference": F.expr("sales_amount - (price * quantity)"),
    },
).select("sales_amount", "price", "quantity", "calculated_revenue", "price_difference")
print(f"Complete Sales Data with Calculated Revenue and Price Difference: {complete_sales_ps.count()}")
complete_sales_ps.show(5)
print(complete_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Complete Sales Data with Calculated Revenue and Price Difference: 100
```

```txt
+------------+------+--------+------------------+------------------+
|sales_amount| price|quantity|calculated_revenue|  price_difference|
+------------+------+--------+------------------+------------------+
|       79.71|200.71|       8|           1605.68|          -1525.97|
|      314.98|257.57|       9|           2318.13|          -2003.15|
|      146.97|257.57|       7|           1802.99|          -1656.02|
|       199.0|257.57|       5|           1287.85|          -1088.85|
|      498.95|257.57|       5|           1287.85|-788.8999999999999|
+------------+------+--------+------------------+------------------+
only showing top 5 rows
```

|      | sales_amount |  price | quantity | calculated_revenue | price_difference |
| ---: | -----------: | -----: | -------: | -----------------: | ---------------: |
|    0 |        48.45 |  15.31 |        8 |             122.48 |           -74.03 |
|    1 |        79.71 | 200.71 |        8 |            1605.68 |         -1525.97 |
|    2 |       314.98 | 257.57 |        9 |            2318.13 |         -2003.15 |
|    3 |          199 | 257.57 |        5 |            1287.85 |         -1088.85 |
|    4 |       356.58 | 200.71 |        5 |            1003.55 |          -646.97 |

</div>

## 4. Window Functions

Window functions are a powerful feature in Pandas that allow us to perform calculations across a set of rows related to the current row. This is particularly useful for time series data, where we may want to calculate rolling averages, cumulative sums, or other metrics based on previous or subsequent rows.

To understand more about the nuances of the window functions, check out some of these guides:

- [Analyzing data with window functions][analysing-window-functions]
- [SQL Window Functions Visualized][visualising-window-functions]

In this section, we will demonstrate how to use window functions to analyze sales data over time. We will start by converting the `date` column to a datetime type, which is necessary for time-based calculations. We will then group the data by date and calculate the total sales for each day.

The first thing that we will do is to group the sales data by date and calculate the total sales for each day. This will give us a daily summary of sales, which we can then use to analyze trends over time.

### PySpark

In PySpark, we can use the [`.groupBy()`][pyspark-groupby] method to group the data by the `date` column, followed by the [`.agg()`][pyspark-groupby-agg] method to calculate the total sales for each day. This will then set us up for further time-based calculations in the following steps.

```python {.pyspark linenums="1" title="Time-based window function"}
df_sales_ps: psDataFrame = df_sales_ps.withColumn("date", F.to_date(df_sales_ps["date"]))
daily_sales_ps: psDataFrame = (
    df_sales_ps.groupBy("date").agg(F.sum("sales_amount").alias("total_sales")).orderBy("date")
)
print(f"Daily Sales Summary: {daily_sales_ps.count()}")
daily_sales_ps.show(5)
print(daily_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales Summary: 100
```

```txt
+----------+-----------+
|      date|total_sales|
+----------+-----------+
|2023-01-01|     490.76|
|2023-01-02|     453.94|
|2023-01-03|     994.51|
|2023-01-04|     184.17|
|2023-01-05|      27.89|
+----------+-----------+
only showing top 5 rows
```

|      | date       | total_sales |
| ---: | :--------- | ----------: |
|    0 | 2023-01-01 |      490.76 |
|    1 | 2023-01-02 |      453.94 |
|    2 | 2023-01-03 |      994.51 |
|    3 | 2023-01-04 |      184.17 |
|    4 | 2023-01-05 |       27.89 |

</div>

Next, we will calculate the lag and lead values for the sales amount. This allows us to compare the current day's sales with the previous and next days' sales.

### PySpark

In PySpark, we can use the [`.lag()`][pyspark-lag] and [`.lead()`][pyspark-lead] functions to calculate the lag and lead values for the sales amount. These functions are used in conjunction with a window specification that defines the order of the rows.

Note that in PySpark, we can define a Window function in one of two ways: using the PySpark API or using the Spark SQL API.

1. **The PySpark API**: The PySpark API allows us to define a window specification using the [`Window()`][pyspark-window] class, which provides methods to specify the ordering of the rows. We can then use the `F.lag()` and `F.lead()` functions to calculate the lag and lead values _over_ a given window on the table.
2. **The Spark SQL API**: The Spark SQL API is used through the [`F.expr()`][pyspark-expr] function, which allows us to write SQL-like expressions for the calculations. This is similar to how we would write SQL queries, but it is executed within the PySpark context.

Here in the below example, we show how the previous day sales can be calculated using the [`.lag()`][pyspark-lag] function in the PySpark API, and the next day sales can be calculated using the [`LEAD()`][sparksql-lead] function in the Spark SQL API. Functionally, both of these two methods achieve the same result, but aesthetically they use slightly different syntax. It is primarily a matter of preference which one you choose to use.

```python {.pyspark linenums="1" title="Calculate lag and lead"}
window_spec_ps: Window = Window.orderBy("date")
daily_sales_ps: psDataFrame = daily_sales_ps.withColumns(
    {
        "previous_day_sales": F.lag("total_sales").over(window_spec_ps),
        "next_day_sales": F.expr("LEAD(total_sales) OVER (ORDER BY date)"),
    },
)
print(f"Daily Sales with Lag and Lead: {daily_sales_ps.count()}")
daily_sales_ps.show(5)
print(daily_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales with Lag and Lead: 100
```

```txt
+----------+-----------+------------------+--------------+
|      date|total_sales|previous_day_sales|next_day_sales|
+----------+-----------+------------------+--------------+
|2023-01-01|     490.76|              NULL|        453.94|
|2023-01-02|     453.94|            490.76|        994.51|
|2023-01-03|     994.51|            453.94|        184.17|
|2023-01-04|     184.17|            994.51|         27.89|
|2023-01-05|      27.89|            184.17|        498.95|
+----------+-----------+------------------+--------------+
only showing top 5 rows
```

|      | date       | total_sales | previous_day_sales | next_day_sales |
| ---: | :--------- | ----------: | -----------------: | -------------: |
|    0 | 2023-01-01 |      490.76 |                nan |         453.94 |
|    1 | 2023-01-02 |      453.94 |             490.76 |         994.51 |
|    2 | 2023-01-03 |      994.51 |             453.94 |         184.17 |
|    3 | 2023-01-04 |      184.17 |             994.51 |          27.89 |
|    4 | 2023-01-05 |       27.89 |             184.17 |         498.95 |

</div>

Now, we can calculate the day-over-day change in sales. This is done by subtracting the previous day's sales from the current day's sales. Then secondly, we can calculate the percentage change in sales using the formula:

```txt
((current_day_sales - previous_day_sales) / previous_day_sales) * 100
```

### PySpark

In PySpark, we can calculate the day-over-day change in sales by subtracting the `previous_day_sales` column from the `total_sales` column. We can also calculate the percentage change in sales using the formula:

```txt
((current_day_sales - previous_day_sales) / previous_day_sales) * 100
```

Here, we have again shown these calculations using two different methods: using the PySpark API and using the Spark SQL API. Realistically, the results for  both of them can be achieved using either method.

```python {.pyspark linenums="1" title="Calculate day-over-day change"}
daily_sales_ps: psDataFrame = daily_sales_ps.withColumns(
    {
        "day_over_day_change": F.col("total_sales") - F.col("previous_day_sales"),
        "pct_change": F.expr("((total_sales - previous_day_sales) / previous_day_sales) * 100").alias("pct_change"),
    }
)
print(f"Daily Sales with Day-over-Day Change: {daily_sales_ps.count()}")
daily_sales_ps.show(5)
print(daily_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales with Day-over-Day Change: 100
```

```txt
+----------+-----------+------------------+--------------+-------------------+------------------+
|      date|total_sales|previous_day_sales|next_day_sales|day_over_day_change|        pct_change|
+----------+-----------+------------------+--------------+-------------------+------------------+
|2023-01-01|     490.76|              NULL|        453.94|               NULL|              NULL|
|2023-01-02|     453.94|            490.76|        994.51| -36.81999999999999|-7.502648952644875|
|2023-01-03|     994.51|            453.94|        184.17|  540.5699999999999|119.08401991452612|
|2023-01-04|     184.17|            994.51|         27.89|            -810.34|-81.48133251551015|
|2023-01-05|      27.89|            184.17|        498.95|-156.27999999999997|-84.85638268990606|
+----------+-----------+------------------+--------------+-------------------+------------------+
only showing top 5 rows
```

|      | date       | total_sales | previous_day_sales | next_day_sales | day_over_day_change | pct_change |
| ---: | :--------- | ----------: | -----------------: | -------------: | ------------------: | ---------: |
|    0 | 2023-01-01 |      490.76 |                nan |         453.94 |                 nan |        nan |
|    1 | 2023-01-02 |      453.94 |             490.76 |         994.51 |              -36.82 |   -7.50265 |
|    2 | 2023-01-03 |      994.51 |             453.94 |         184.17 |              540.57 |    119.084 |
|    3 | 2023-01-04 |      184.17 |             994.51 |          27.89 |             -810.34 |   -81.4813 |
|    4 | 2023-01-05 |       27.89 |             184.17 |         498.95 |             -156.28 |   -84.8564 |

</div>

Next, we will calculate the rolling average of sales over a 7-day window. Rolling averages (aka moving averages) are useful for smoothing out short-term fluctuations and highlighting longer-term trends in the data. This is particularly useful in time series analysis, where we want to understand the underlying trend in the data without being overly influenced by short-term variations. It is also a very common technique used in financial analysis to analyze stock prices, sales data, and other time series data.

### PySpark

In PySpark, we can calculate the 7-day moving average of sales using the [`F.avg()`][pyspark-avg] function in combination with the [`Window()`][pyspark-window] class. The [`Window()`][pyspark-window] class allows us to define a window specification for the calculation. We can use the [`.orderBy()`][pyspark-window-orderby] method to specify the order of the rows in the window, and the [`.rowsBetween()`][pyspark-window-rowsbetween] method to specify the range of rows to include in the window. The [`F.avg()`][pyspark-avg] function is then able to calculate the average of the `total_sales` column over the specified window.

As with many aspects of PySpark, there are multiple ways to achieve the same result. In this case, we can use either the [`F.avg()`][pyspark-avg] function with the [`Window()`][pyspark-window] class, or we can use the SQL expression syntax with the [`F.expr()`][pyspark-expr] function. Both methods will yield the same result.

```python {.pyspark linenums="1" title="Calculate 7-day moving average"}
daily_sales_ps: psDataFrame = daily_sales_ps.withColumns(
    {
        "7d_moving_avg": F.avg("total_sales").over(Window.orderBy("date").rowsBetween(-6, 0)),
        "7d_rolling_avg": F.expr("AVG(total_sales) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)"),
    }
)
print(f"Daily Sales with 7-Day Moving Average: {daily_sales_ps.count()}")
daily_sales_ps.show(5)
print(daily_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales with 7-Day Moving Average: 100
```

```txt
+----------+-----------+------------------+--------------+-------------------+------------------+-----------------+-----------------+
|      date|total_sales|previous_day_sales|next_day_sales|day_over_day_change|        pct_change|    7d_moving_avg|   7d_rolling_avg|
+----------+-----------+------------------+--------------+-------------------+------------------+-----------------+-----------------+
|2023-01-01|     490.76|              NULL|        453.94|               NULL|              NULL|           490.76|           490.76|
|2023-01-02|     453.94|            490.76|        994.51| -36.81999999999999|-7.502648952644875|           472.35|           472.35|
|2023-01-03|     994.51|            453.94|        184.17|  540.5699999999999|119.08401991452612|646.4033333333333|646.4033333333333|
|2023-01-04|     184.17|            994.51|         27.89|            -810.34|-81.48133251551015|          530.845|          530.845|
|2023-01-05|      27.89|            184.17|        498.95|-156.27999999999997|-84.85638268990606|          430.254|          430.254|
+----------+-----------+------------------+--------------+-------------------+------------------+-----------------+-----------------+
only showing top 5 rows
```

|      | date       | total_sales | previous_day_sales | next_day_sales | day_over_day_change | pct_change | 7d_moving_avg | 7d_rolling_avg |
| ---: | :--------- | ----------: | -----------------: | -------------: | ------------------: | ---------: | ------------: | -------------: |
|    0 | 2023-01-01 |      490.76 |                nan |         453.94 |                 nan |        nan |        490.76 |         490.76 |
|    1 | 2023-01-02 |      453.94 |             490.76 |         994.51 |              -36.82 |   -7.50265 |        472.35 |         472.35 |
|    2 | 2023-01-03 |      994.51 |             453.94 |         184.17 |              540.57 |    119.084 |       646.403 |        646.403 |
|    3 | 2023-01-04 |      184.17 |             994.51 |          27.89 |             -810.34 |   -81.4813 |       530.845 |        530.845 |
|    4 | 2023-01-05 |       27.89 |             184.17 |         498.95 |             -156.28 |   -84.8564 |       430.254 |        430.254 |

</div>

Finally, we can visualize the daily sales data along with the 7-day moving average using Plotly. This allows us to see the trends in sales over time and how the moving average smooths out the fluctuations in daily sales.

For this, we will again utilise [Plotly][plotly] to create an interactive line chart that displays both the daily sales and the 7-day moving average. The chart will have the date on the x-axis and the sales amount on the y-axis, with two lines representing the daily sales and the moving average.

The graph will be [instantiated][python-class-instantiation] using the [`go.Figure()`][plotly-figure] class, and using the [`.add_trace()`][plotly-add-traces] method we will add two traces to the figure: one for the daily sales and one for the 7-day moving average. The [`go.Scatter()`][plotly-scatter] class is used to create the line traces, by defining `mode="lines"` to display the data as a line chart.

Finally, we will use the [`.update_layout()`][plotly-update_layout] method to set the titles for the chart, and the position of the legend.

### PySpark

Plotly is not able to interpret PySpark DataFrames directly, so we need to convert the PySpark DataFrame to a Pandas DataFrame before plotting. This can be done using the [`.toPandas()`][pyspark-topandas] method. We can then parse the columns from the Pandas DataFrame to create the traces for the daily sales and the 7-day moving average.

```python {.pyspark linenums="1" title="Plot results"}
fig: go.Figure = (
    go.Figure()
    .add_trace(
        go.Scatter(
            x=daily_sales_ps.toPandas()["date"],
            y=daily_sales_ps.toPandas()["total_sales"],
            mode="lines",
            name="Daily Sales",
        )
    )
    .add_trace(
        go.Scatter(
            x=daily_sales_ps.toPandas()["date"],
            y=daily_sales_ps.toPandas()["7d_moving_avg"],
            mode="lines",
            name="7-Day Moving Average",
            line_width=3,
        ),
    )
    .update_layout(
        title="Daily Sales with 7-Day Moving Average",
        xaxis_title="Date",
        yaxis_title="Sales Amount ($)",
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1,
    )
)
fig.write_html("images/pt4_daily_sales_with_7d_avg_ps.html", include_plotlyjs="cdn", full_html=True)
fig.show()
```

<div class="result" markdown>


</div>



## 5. Ranking and Partitioning

The fifth section will demonstrate how to rank and partition data. This is useful for identifying top performers, such as the highest spending customers or the most popular products.

### PySpark

In PySpark, we can use the [`F.dense_rank()`][pyspark-dense_rank] function in combination with the [`Window()`][pyspark-window] class to rank values in a DataFrame. The [`Window()`][pyspark-window] class allows us to define a window specification for the calculation, and the [`F.dense_rank()`][pyspark-dense_rank] function calculates the dense rank of each row within that window.

```python {.pyspark linenums="1" title="Rank customers by total spending"}
customer_spending_ps: psDataFrame = (
    df_sales_ps.groupBy("customer_id")
    .agg(F.sum("sales_amount").alias("total_spending"))
    .withColumn("rank", F.dense_rank().over(Window.orderBy(F.desc("total_spending"))))
    .orderBy("rank")
)
print(f"Customer Spending Summary: {customer_spending_ps.count()}")
customer_spending_ps.show(5)
print(customer_spending_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Customer Spending Summary: 61
```

```txt
+-----------+------------------+----+
|customer_id|    total_spending|rank|
+-----------+------------------+----+
|         15|           2297.55|   1|
|          4|           2237.49|   2|
|         62|           2177.35|   3|
|         60|2086.0899999999997|   4|
|         21|           2016.95|   5|
+-----------+------------------+----+
```

|      | customer_id | total_spending | rank |
| ---: | ----------: | -------------: | ---: |
|    0 |          15 |        2297.55 |    1 |
|    1 |           4 |        2237.49 |    2 |
|    2 |          62 |        2177.35 |    3 |
|    3 |          60 |        2086.09 |    4 |
|    4 |          21 |        2016.95 |    5 |

</div>

Next, we will rank products based on the quantity sold, partitioned by the product category. This will help us identify the most popular products within each category.

### PySpark

In PySpark, we can use the [`F.dense_rank()`][pyspark-dense_rank] function in combination with the [`Window()`][pyspark-window] class to rank products within each category based on the total quantity sold. We can define the partitioning by using the [`.partitionBy()`][pyspark-window-partitionby] method and parse'ing in the `"category"` column. We can then define the ordering by using the [`.orderBy()`][pyspark-window-orderby] method and parse'ing in the `"total_quantity"` expression to order the products by total quantity sold in descending order with the [`F.desc()`][pyspark-desc] method.

Here, we have also provided an alternative way to define the rank by using the Spark SQL method. The outcome is the same, it's simply written in a SQL-like expression.

```python {.pyspark linenums="1" title="Rank products by quantity sold, by category"}
product_popularity_ps: psDataFrame = (
    df_sales_ps.groupBy("category", "product_id")
    .agg(F.sum("quantity").alias("total_quantity"))
    .withColumns(
        {
            "rank_p": F.dense_rank().over(Window.partitionBy("category").orderBy(F.desc("total_quantity"))),
            "rank_s": F.expr("DENSE_RANK() OVER (PARTITION BY category ORDER BY total_quantity DESC)"),
        }
    )
    .orderBy("rank_p")
)
print(f"Product Popularity Summary: {product_popularity_ps.count()}")
product_popularity_ps.show(10)
print(product_popularity_ps.limit(10).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Product Popularity Summary: 78
```

```txt
+-----------+----------+--------------+------+------+
|   category|product_id|total_quantity|rank_p|rank_s|
+-----------+----------+--------------+------+------+
|   Clothing|         7|             9|     1|     1|
|      Books|        11|            14|     1|     1|
|Electronics|        37|            16|     1|     1|
|       Food|        45|            34|     1|     1|
|       Home|         3|            10|     1|     1|
|      Books|        28|             9|     2|     2|
|Electronics|        35|            11|     2|     2|
|       Home|        29|             5|     2|     2|
|       Home|        48|             5|     2|     2|
|       Home|         9|             5|     2|     2|
+-----------+----------+--------------+------+------+
only showing top 10 rows
```

|      | category    | product_id | total_quantity | rank_p | rank_s |
| ---: | :---------- | ---------: | -------------: | -----: | -----: |
|    0 | Clothing    |          7 |              9 |      1 |      1 |
|    1 | Books       |         11 |             14 |      1 |      1 |
|    2 | Electronics |         37 |             16 |      1 |      1 |
|    3 | Food        |         45 |             34 |      1 |      1 |
|    4 | Home        |          3 |             10 |      1 |      1 |
|    5 | Books       |         28 |              9 |      2 |      2 |
|    6 | Clothing    |         35 |              8 |      2 |      2 |
|    7 | Electronics |         35 |             11 |      2 |      2 |
|    8 | Food        |          1 |             16 |      2 |      2 |
|    9 | Home        |         29 |              5 |      2 |      2 |

</div>

## Conclusion

This comprehensive guide has demonstrated how to perform essential data querying and manipulation operations across four powerful tools: [Pandas], [SQL][sqlite], [PySpark], and [Polars]. Each tool brings unique advantages to the data processing landscape, and understanding their strengths helps you choose the right tool for your specific use case.


### Tool Comparison and Use Cases

<div class="grid cards" markdown>

-   [**Pandas**][pandas] has an extensive ecosystem, making it ideal for:

- Small to medium datasets (up to millions of rows)
- Interactive data exploration and visualization
- Data preprocessing for machine learning workflows
- Quick statistical analysis and reporting

---

**Pandas** remains the go-to choice for exploratory data analysis and rapid prototyping.

-   [**SQL**][sqlite] excels in:

- Working with relational databases and data warehouses
- Complex joins and subqueries
- Declarative data transformations
- Team environments where SQL knowledge is widespread

---

**SQL** provides the universal language of data with unmatched expressiveness for complex queries

-   [**PySpark**][pyspark] is great for when you need:

- Processing datasets that don't fit in memory (terabytes or larger)
- Distributed computing across clusters
- Integration with Hadoop ecosystem components
- Scalable machine learning with MLlib

---

**PySpark** unlocks the power of distributed computing for big data scenarios.

-   [**Polars**][polars] is particularly valuable for:

- Large datasets that require fast processing (gigabytes to small terabytes)
- Performance-critical applications
- Memory-constrained environments
- Lazy evaluation and query optimization

---

**Polars** emerges as the high-performance alternative with excellent memory efficiency.

</div>


### Key Techniques Covered

Throughout this guide, we've explored fundamental data manipulation patterns that remain consistent across all tools:

1. **Data Filtering and Selection** - Essential for subsetting data based on conditions
2. **Grouping and Aggregation** - Critical for summarizing data by categories
3. **Joining and Merging** - Necessary for combining data from multiple sources
4. **Window Functions** - Powerful for time-series analysis and advanced calculations
5. **Ranking and Partitioning** - Useful for identifying top performers and comparative analysis


### Best Practices and Recommendations

When working with any of these tools, consider these best practices:

- **Start with the right tool**: Match your tool choice to your data size, infrastructure, and team expertise
- **Understand your data**: Always examine data types, null values, and distributions before processing
- **Optimize for readability**: Write clear, well-documented code that your future self and teammates can understand
- **Profile performance**: Measure execution time and memory usage, especially for large datasets
- **Leverage built-in optimizations**: Use vectorized operations, avoid loops, and take advantage of lazy evaluation where available


### Moving Forward

The data landscape continues to evolve rapidly, with new tools and techniques emerging regularly. The fundamental concepts demonstrated in this guide—filtering, grouping, joining, and analytical functions—remain constant across platforms. By mastering these core concepts, you'll be well-equipped to adapt to new tools and technologies as they arise.

Whether you're analyzing customer behavior, processing sensor data, or building machine learning models, the techniques in this guide provide a solid foundation for effective data manipulation. Remember that the best tool is often the one that best fits your specific requirements for performance, scalability, and team capabilities.

Continue practicing with real datasets, explore advanced features of each tool, and stay curious about emerging technologies in the data processing ecosystem. The skills you've learned here will serve as building blocks for increasingly sophisticated data analysis and engineering tasks.


<!--
-- ------------------------ --
--  Shortcuts & Hyperlinks  --
-- ------------------------ --
-->

<!-- Python -->
[python-print]: https://docs.python.org/3/library/functions.html#print
[python-class-instantiation]: https://docs.python.org/3/tutorial/classes.html#:~:text=example%20class%22.-,Class%20instantiation,-uses%20function%20notation
[numpy]: https://numpy.org/

<!-- Storage -->
[hdfs]: https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html
[s3]: https://aws.amazon.com/s3/
[adls]: https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction
[jdbc]: https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html

<!-- Guides -->
[analysing-window-functions]: https://docs.snowflake.com/en/user-guide/functions-window-using
[visualising-window-functions]: https://medium.com/learning-sql/sql-window-function-visualized-fff1927f00f2
[querying-data]: https://github.com/data-science-extensions/data-science-extensions/blob/main/docs/guides/querying-data/index.md

<!-- Pandas -->
[pandas]: https://pandas.pydata.org/
[pandas-head]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html
[pandas-read_sql]: https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html
[pandas-subsetting]: https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
[pandas-agg]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html
[pandas-groupby]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
[pandas-groupby-agg]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.agg.html
[pandas-groupby-sum]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.sum.html
[pandas-columns]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html
[pandas-rename]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
[pandas-reset_index]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html
[pandas-merge]: https://pandas.pydata.org/docs/reference/api/pandas.merge.html
[pandas-shift]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html
[pandas-sort_values]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
[pandas-pct_change]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html
[pandas-rolling]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
[pandas-rolling-mean]: https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.mean.html
[pandas-rank]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html
[pandas-assign]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html

<!-- SQL -->
[postgresql]: https://www.postgresql.org/
[mysql]: https://www.mysql.com/
[t-sql]: https://learn.microsoft.com/en-us/sql/t-sql/
[pl-sql]: https://www.oracle.com/au/database/technologies/appdev/plsql.html
[sql-wiki]: https://en.wikipedia.org/wiki/SQL
[sql-iso]: https://www.iso.org/standard/76583.html
[sqlite]: https://sqlite.org/
[sqlite3]: https://docs.python.org/3/library/sqlite3.html
[sqlite3-connect]: https://docs.python.org/3/library/sqlite3.html#sqlite3.connect
[sqlite-where]: https://sqlite.org/lang_select.html#whereclause
[sqlite-select]: https://sqlite.org/lang_select.html
[sqlite-tutorial-join]: https://www.sqlitetutorial.net/sqlite-join/

<!-- SQL -->
[spark-sql]: https://spark.apache.org/sql/
[pyspark]: https://spark.apache.org/docs/latest/api/python/
[pyspark-sparksession]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.html
[pyspark-builder]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.html#pyspark.sql.SparkSession.builder
[pyspark-getorcreate]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.builder.getOrCreate.html
[pyspark-createdataframe]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.createDataFrame.html
[pyspark-show]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.show.html
[pyspark-create-dataframe-from-dict]: https://sparkbyexamples.com/pyspark/pyspark-create-dataframe-from-dictionary/
[pyspark-filter]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.filter.html
[pyspark-where]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.where.html
[pyspark-filtering]: https://sparkbyexamples.com/pyspark/pyspark-where-filter/
[pyspark-select]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.select.html
[pyspark-agg]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.agg.html
[pyspark-groupby]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.groupBy.html
[pyspark-groupby-agg]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.agg.html
[pyspark-withcolumnsrenamed]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.withColumnsRenamed.html
[pyspark-topandas]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.toPandas.html
[pyspark-join]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.join.html
[pyspark-withcolumns]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.withColumns.html
[pyspark-col]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.col.html
[pyspark-expr]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.expr.html
[pyspark-lead]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.lead.html
[pyspark-lag]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.lag.html
[sparksql-lead]: https://spark.apache.org/docs/latest/api/sql/index.html#lead
[pyspark-window]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.Window.html
[pyspark-avg]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.avg.html
[pyspark-window-orderby]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.Window.orderBy.html
[pyspark-window-partitionby]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.Window.partitionBy.html
[pyspark-window-rowsbetween]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.Window.rowsBetween.html
[pyspark-dense_rank]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.dense_rank.html
[pyspark-desc]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.desc.html

<!-- Polars -->
[polars]: https://www.pola.rs/
[polars-head]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.head.html
[polars-filter]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.filter.html
[polars-filtering]: https://docs.pola.rs/user-guide/transformations/time-series/filter/
[polars-col]: https://docs.pola.rs/api/python/stable/reference/expressions/col.html
[polars-select]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.select.html
[polars-groupby]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.group_by.html
[polars-groupby-agg]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.dataframe.group_by.GroupBy.agg.html
[polars-groupby-sum]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.dataframe.group_by.GroupBy.sum.html
[polars-rename]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.rename.html
[polars-join]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join.html
[polars-with-columns]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.with_columns.html
[polars-shift]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.shift.html
[polars-sort]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.sort.html
[polars-rolling-mean]: https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.rolling_mean.html
[polars-rank]: https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.rank.html
[polars-over]: https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.over.html

<!-- Plotly -->
[plotly]: https://plotly.com/python/
[plotly-express]: https://plotly.com/python/plotly-express/
[plotly-bar]: https://plotly.com/python/bar-charts/
[plotly-figure]: https://plotly.com/python/creating-and-updating-figures/#figures-as-graph-objects
[plotly-add-traces]: https://plotly.com/python/creating-and-updating-figures/#adding-traces
[plotly-scatter]: https://plotly.com/python/line-and-scatter/
[plotly-update_layout]: https://plotly.com/python/reference/layout/
