# Data Querying for Data Scientists

### A Comprehensive Guide of using Pandas, SQL, PySpark, and Polars for Data Manipulation Techniques, with Practical Examples and Visualisations


## Introduction

Working as a Data Scientist or Data Engineer often involves querying data from various sources. There are many tools and libraries available to perform these tasks, each with its own strengths and weaknesses. Also, there are many different ways to achieve similar results, depending on the tool or library used. It's important to be familiar with these different methods to choose the best one for your specific use case.

This article provides a comprehensive guide on how to query data using different tools and libraries, including Pandas, SQL, PySpark, and Polars. Each section will cover the setup, data creation, and various querying techniques such as filtering, grouping, joining, window functions, ranking, and sorting. The output will be identical across all tools, but the transformations will be implemented using the specific syntax and features of each library. Therefore allowing you to compare the different approaches and understand the nuances of each method.


## Overview of the Different Libraries

Before we dive into the querying techniques, let's take a moment to understand the different libraries and tools we will be using in this article. Each library has its own strengths and weaknesses, and understanding these can help you choose the right tool for your specific use case.

Throughout this article, you can easily switch between the different libraries by selecting the appropriate tab. Each section will provide the same functionality, but implemented using the specific syntax and features of each library.

### Pandas

[Pandas] is a powerful data manipulation library in Python that provides data structures and functions for working with structured data. It is widely used for data analysis and manipulation tasks.

Historically, Pandas was one of the first libraries to provide a DataFrame structure, which is similar to a table in a relational database. It allows for easy data manipulation, filtering, grouping, and aggregation. Pandas is built on top of [NumPy][numpy] and provides a high-level interface for working with data. It is particularly well-suited for small to medium-sized datasets and is often used in Data Science and Machine Learning workflows.

Pandas provides a rich set of functionalities for data manipulation, including filtering, grouping, joining, and window functions. It also integrates well with other libraries such as Matplotlib and Seaborn for data visualization, making it a popular choice among data scientists and analysts.

While Pandas is both powerful and popular, it is important to note that it operates **in-memory**, which means that it may not be suitable for very large datasets that do not fit into memory. In such cases, other libraries like PySpark or Polars may be more appropriate.

### SQL

[SQL][sql-wiki] (Structured Query Language) is a standard language for managing and manipulating relational databases. It is widely used for querying and modifying data in databases. SQL is a declarative language, meaning that you specify what you want to retrieve or manipulate without detailing how to do it. This makes SQL queries concise and expressive. SQL is particularly well-suited for working with large datasets and complex queries. It provides powerful features for filtering, grouping, joining, and aggregating data. SQL is the backbone of many database systems.

SQL is actually a language (like Python is a language), not a library (like Pandas is a library), and it is used to interact with relational databases. The core of the SQL language is actually an [ISO standard][sql-iso], which means that the basic syntax and functionality are consistent across different database systems. However, each database system may have its own extensions or variations of SQL, which can lead to differences in syntax and features. Each database system can be considered as variations (or dialects) of SQL, with their own specific features and optimizations and syntax ehnancements.

Some of the more popular SQL dialects include:

<div class="mdx-three-columns" markdown>

- [SQLite]
- [PostgreSQL]
- [MySQL]
- [Spark SQL][spark-sql]
- [SQL Server (t-SQL)][t-sql]
- [Oracle SQL (pl-SQL)][pl-sql]

</div>

### PySpark

[PySpark] is the Python API for Apache Spark, a distributed computing framework that allows for large-scale data processing. PySpark provides a high-level interface for working with Spark, making it easier to write distributed data processing applications in Python. It is particularly well-suited for big data processing and analytics.

PySpark provides a DataFrame API similar to Pandas, but it is designed to work with large datasets that do not fit into memory. It allows for distributed data processing across a cluster of machines, making it suitable for big data applications. PySpark supports various data sources, including [HDFS], [S3], [ADLS], and [JDBC], and provides powerful features for filtering, grouping, joining, and aggregating data.

While PySpark is a powerful tool for big data processing, it can be more complex to set up and use compared to Pandas. It requires a Spark cluster and may have a steeper learning curve for those unfamiliar with distributed computing concepts. However, it is an excellent choice for processing large datasets and performing complex data transformations.

### Polars

[Polars] is a fast DataFrame library for Python that is designed for high-performance data manipulation. It is built on top of Rust and provides a DataFrame API similar to Pandas, but with a focus on performance and memory efficiency. Polars is particularly well-suited for large datasets and complex queries.

Polars supports lazy evaluation, which allows for optimizations in query execution. Polars also provides powerful features for filtering, grouping, joining, and aggregating data, making it a great choice for data analysis tasks.

While Polars is a relatively new library compared to Pandas, it has gained popularity for its performance and ease of use. It is designed to be a drop-in replacement for Pandas, allowing users to leverage its performance benefits without significant changes to their existing code. It is particularly useful for data scientists and analysts who need to work with large datasets and require fast data manipulation capabilities. The setup is simple and straightforward, similar to Pandas, and less complex than PySpark. It is a great choice for data analysis tasks that require high performance and memory efficiency.


## Setup

Before we start querying data, we need to set up our environment. This includes importing the necessary libraries, creating sample data, and defining constants that will be used throughout the article. The following sections will guide you through this setup process. The code for this article is also available on GitHub: [querying-data](...) {==**UPDATE URL**==}.

### Pandas

```python {.pandas linenums="1" title="Setup"}
# StdLib Imports
from typing import Any

# Third Party Imports
import numpy as np
import pandas as pd
from plotly import express as px, graph_objects as go

# Set seed for reproducibility
np.random.seed(42)

# Determine the number of records to generate
n_records = 100
```

### SQL

```python {.sql linenums="1" title="Setup"}
# StdLib Imports
import sqlite3
from typing import Any

# Third Party Imports
import numpy as np
import pandas as pd
from plotly import express as px, graph_objects as go
```

### PySpark

```python {.pyspark linenums="1" title="Setup"}
# StdLib Imports
from typing import Any

# Third Party Imports
import numpy as np
from plotly import express as px, graph_objects as go
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
```

### Polars

```python {.polars linenums="1" title="Setup"}
# StdLib Imports
from typing import Any

# Third Party Imports
import numpy as np
import polars as pl
from plotly import express as px, graph_objects as go

# Set seed for reproducibility
np.random.seed(42)

# Determine the number of records to generate
n_records = 100
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
# Create product dimension table
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
# Create customer dimension table
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

### Pandas

To create the dataframes in Pandas, we will use the data we generated earlier. We will parse the dictionaries into Pandas DataFrames, which will allow us to perform various data manipulation tasks.

```python {.pandas linenums="1" title="Create DataFrames"}
df_sales_pd: pd.DataFrame = pd.DataFrame(sales_data)
df_product_pd: pd.DataFrame = pd.DataFrame(product_data)
df_customer_pd: pd.DataFrame = pd.DataFrame(customer_data)
```

Once the data is created, we can check that it has been loaded correctly by displaying the first few rows of each DataFrame. To do this, we will use the [`.head()`][pandas-head] method to display the first 5 rows of each DataFrame, and then parse to the [`print()`][python-print] function to display the DataFrame in a readable format.

```python {.pandas linenums="1" title="Check Sales DataFrame"}
print(f"Sales DataFrame: {len(df_sales_pd)}")
print(df_sales_pd.head(5))
print(df_sales_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Sales DataFrame: 100
```

```txt
        date  customer_id  product_id     category  sales_amount  quantity
0 2023-01-01           52          45         Food        490.76         7
1 2023-01-02           93          41  Electronics        453.94         5
2 2023-01-03           15          29         Home        994.51         5
3 2023-01-04           72          15  Electronics        184.17         7
4 2023-01-05           61          45         Food         27.89         9
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-01 00:00:00 |          52 |         45 | Food        |       490.76 |        7 |
|    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
|    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
|    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
|    4 | 2023-01-05 00:00:00 |          61 |         45 | Food        |        27.89 |        9 |

</div>

```python {.pandas linenums="1" title="Check Product DataFrame"}
print(f"Product DataFrame: {len(df_product_pd)}")
print(df_product_pd.head(5))
print(df_product_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Product DataFrame: 50
```

```txt
   product_id product_name   price  category  supplier_id
0           1    Product 1  257.57      Food            8
1           2    Product 2  414.96  Clothing            5
2           3    Product 3  166.82  Clothing            8
3           4    Product 4  448.81      Food            4
4           5    Product 5  200.71      Food            8
```

|      | product_id | product_name |  price | category | supplier_id |
| ---: | ---------: | :----------- | -----: | :------- | ----------: |
|    0 |          1 | Product 1    | 257.57 | Food     |           8 |
|    1 |          2 | Product 2    | 414.96 | Clothing |           5 |
|    2 |          3 | Product 3    | 166.82 | Clothing |           8 |
|    3 |          4 | Product 4    | 448.81 | Food     |           4 |
|    4 |          5 | Product 5    | 200.71 | Food     |           8 |

</div>

```python {.pandas linenums="1" title="Check Customer DataFrame"}
print(f"Customer DataFrame: {len(df_customer_pd)}")
print(df_customer_pd.head(5))
print(df_customer_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Customer DataFrame: 100
```

```txt
   customer_id customer_name         city state      segment
0            1    Customer 1      Phoenix    NY    Corporate
1            2    Customer 2      Phoenix    CA  Home Office
2            3    Customer 3      Phoenix    NY  Home Office
3            4    Customer 4  Los Angeles    NY     Consumer
4            5    Customer 5  Los Angeles    IL  Home Office
```

|      | customer_id | customer_name | city        | state | segment     |
| ---: | ----------: | :------------ | :---------- | :---- | :---------- |
|    0 |           1 | Customer 1    | Phoenix     | NY    | Corporate   |
|    1 |           2 | Customer 2    | Phoenix     | CA    | Home Office |
|    2 |           3 | Customer 3    | Phoenix     | NY    | Home Office |
|    3 |           4 | Customer 4    | Los Angeles | NY    | Consumer    |
|    4 |           5 | Customer 5    | Los Angeles | IL    | Home Office |

</div>

### SQL

To create the dataframes in SQL, we will use the data we generated earlier. Firstly, we need to create the SQLite database. This will be an in-memory database for demonstration purposes, but in a real-world scenario, you would typically connect to a persistent (on-disk) database. To do this, we will use the [`sqlite3`][sqlite3] library to create a connection to the database, which we define with the `:memory:` parameter on the [`.connect()`][sqlite3-connect] function. The result is to create a temporary database that exists only during the lifetime of the connection.

Next, we will then parse the dictionaries into Pandas DataFrames, which will then be loaded into an SQLite database. This allows us to perform various data manipulation tasks using SQL queries.

```python {.sql linenums="1" title="Create DataFrames"}
# Creates SQLite database and tables
conn: sqlite3.Connection = sqlite3.connect(":memory:")
pd.DataFrame(sales_data).to_sql("sales", conn, index=False, if_exists="replace")
pd.DataFrame(product_data).to_sql("product", conn, index=False, if_exists="replace")
pd.DataFrame(customer_data).to_sql("customer", conn, index=False, if_exists="replace")
```

Once the data is created, we can check that it has been loaded correctly by displaying the first few rows of each DataFrame. To do this, we will use the [`pd.read_sql()`][pandas-read_sql] function to execute SQL queries and retrieve the data from the database. We will then parse the results to the [`print()`][python-print] function to display the DataFrame in a readable format.

```python {.sql linenums="1" title="Check Sales DataFrame"}
print(f"Sales Table: {len(pd.read_sql('SELECT * FROM sales', conn))}")
print(pd.read_sql("SELECT * FROM sales LIMIT 5", conn))
print(pd.read_sql("SELECT * FROM sales LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt
Sales Table: 100
```

```txt
                  date  customer_id  product_id     category  sales_amount  quantity
0  2023-01-01 00:00:00           52          45         Food        490.76         7
1  2023-01-02 00:00:00           93          41  Electronics        453.94         5
2  2023-01-03 00:00:00           15          29         Home        994.51         5
3  2023-01-04 00:00:00           72          15  Electronics        184.17         7
4  2023-01-05 00:00:00           61          45         Food         27.89         9
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-01 00:00:00 |          52 |         45 | Food        |       490.76 |        7 |
|    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
|    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
|    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
|    4 | 2023-01-05 00:00:00 |          61 |         45 | Food        |        27.89 |        9 |

</div>

```python {.sql linenums="1" title="Check Product DataFrame"}
print(f"Product Table: {len(pd.read_sql('SELECT * FROM product', conn))}")
print(pd.read_sql("SELECT * FROM product LIMIT 5", conn))
print(pd.read_sql("SELECT * FROM product LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt
Product Table: 50
```

```txt
   product_id product_name   price  category  supplier_id
0           1    Product 1  257.57      Food            8
1           2    Product 2  414.96  Clothing            5
2           3    Product 3  166.82  Clothing            8
3           4    Product 4  448.81      Food            4
4           5    Product 5  200.71      Food            8
```

|      | product_id | product_name |  price | category | supplier_id |
| ---: | ---------: | :----------- | -----: | :------- | ----------: |
|    0 |          1 | Product 1    | 257.57 | Food     |           8 |
|    1 |          2 | Product 2    | 414.96 | Clothing |           5 |
|    2 |          3 | Product 3    | 166.82 | Clothing |           8 |
|    3 |          4 | Product 4    | 448.81 | Food     |           4 |
|    4 |          5 | Product 5    | 200.71 | Food     |           8 |

</div>

```python {.sql linenums="1" title="Check Customer DataFrame"}
print(f"Customer Table: {len(pd.read_sql('SELECT * FROM customer', conn))}")
print(pd.read_sql("SELECT * FROM customer LIMIT 5", conn))
print(pd.read_sql("SELECT * FROM customer LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt
Customer Table: 100
```

```txt
   customer_id customer_name         city state      segment
0            1    Customer 1      Phoenix    NY    Corporate
1            2    Customer 2      Phoenix    CA  Home Office
2            3    Customer 3      Phoenix    NY  Home Office
3            4    Customer 4  Los Angeles    NY     Consumer
4            5    Customer 5  Los Angeles    IL  Home Office
```

|      | customer_id | customer_name | city        | state | segment     |
| ---: | ----------: | :------------ | :---------- | :---- | :---------- |
|    0 |           1 | Customer 1    | Phoenix     | NY    | Corporate   |
|    1 |           2 | Customer 2    | Phoenix     | CA    | Home Office |
|    2 |           3 | Customer 3    | Phoenix     | NY    | Home Office |
|    3 |           4 | Customer 4    | Los Angeles | NY    | Consumer    |
|    4 |           5 | Customer 5    | Los Angeles | IL    | Home Office |

</div>

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

### Polars

To create the dataframes in Polars, we will use the data we generated earlier. We will parse the dictionaries into Polars DataFrames, which will allow us to perform various data manipulation tasks.

```python {.polars linenums="1" title="Create DataFrames"}
df_sales_pl: pl.DataFrame = pl.DataFrame(sales_data)
df_product_pl: pl.DataFrame = pl.DataFrame(product_data)
df_customer_pl: pl.DataFrame = pl.DataFrame(customer_data)
```

Once the data is created, we can check that it has been loaded correctly by displaying the first few rows of each DataFrame. To do this, we will use the [`.head()`][polars-head] method to display the first `5` rows of each DataFrame, and then parse to the [`print()`][python-print] function to display the DataFrame in a readable format.

```python {.polars linenums="1" title="Check Sales DataFrame"}
print(f"Sales DataFrame: {df_sales_pl.shape[0]}")
print(df_sales_pl.head(5))
print(df_sales_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales DataFrame: 100
```

```txt
shape: (5, 6)
┌─────────────────────┬─────────────┬────────────┬─────────────┬──────────────┬──────────┐
│ date                ┆ customer_id ┆ product_id ┆ category    ┆ sales_amount ┆ quantity │
│ ---                 ┆ ---         ┆ ---        ┆ ---         ┆ ---          ┆ ---      │
│ datetime[ns]        ┆ i64         ┆ i64        ┆ str         ┆ f64          ┆ i64      │
╞═════════════════════╪═════════════╪════════════╪═════════════╪══════════════╪══════════╡
│ 2023-01-01 00:00:00 ┆ 52          ┆ 45         ┆ Food        ┆ 490.76       ┆ 7        │
│ 2023-01-02 00:00:00 ┆ 93          ┆ 41         ┆ Electronics ┆ 453.94       ┆ 5        │
│ 2023-01-03 00:00:00 ┆ 15          ┆ 29         ┆ Home        ┆ 994.51       ┆ 5        │
│ 2023-01-04 00:00:00 ┆ 72          ┆ 15         ┆ Electronics ┆ 184.17       ┆ 7        │
│ 2023-01-05 00:00:00 ┆ 61          ┆ 45         ┆ Food        ┆ 27.89        ┆ 9        │
└─────────────────────┴─────────────┴────────────┴─────────────┴──────────────┴──────────┘
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-01 00:00:00 |          52 |         45 | Food        |       490.76 |        7 |
|    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
|    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
|    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
|    4 | 2023-01-05 00:00:00 |          61 |         45 | Food        |        27.89 |        9 |

</div>

```python {.polars linenums="1" title="Check Product DataFrame"}
print(f"Product DataFrame: {df_product_pl.shape[0]}")
print(df_product_pl.head(5))
print(df_product_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Product DataFrame: 50
```

```txt
shape: (5, 5)
┌────────────┬──────────────┬────────┬──────────┬─────────────┐
│ product_id ┆ product_name ┆ price  ┆ category ┆ supplier_id │
│ ---        ┆ ---          ┆ ---    ┆ ---      ┆ ---         │
│ i64        ┆ str          ┆ f64    ┆ str      ┆ i64         │
╞════════════╪══════════════╪════════╪══════════╪═════════════╡
│ 1          ┆ Product 1    ┆ 257.57 ┆ Food     ┆ 8           │
│ 2          ┆ Product 2    ┆ 414.96 ┆ Clothing ┆ 5           │
│ 3          ┆ Product 3    ┆ 166.82 ┆ Clothing ┆ 8           │
│ 4          ┆ Product 4    ┆ 448.81 ┆ Food     ┆ 4           │
│ 5          ┆ Product 5    ┆ 200.71 ┆ Food     ┆ 8           │
└────────────┴──────────────┴────────┴──────────┴─────────────┘
```

|      | product_id | product_name |  price | category | supplier_id |
| ---: | ---------: | :----------- | -----: | :------- | ----------: |
|    0 |          1 | Product 1    | 257.57 | Food     |           8 |
|    1 |          2 | Product 2    | 414.96 | Clothing |           5 |
|    2 |          3 | Product 3    | 166.82 | Clothing |           8 |
|    3 |          4 | Product 4    | 448.81 | Food     |           4 |
|    4 |          5 | Product 5    | 200.71 | Food     |           8 |

</div>

```python {.polars linenums="1" title="Check Customer DataFrame"}
print(f"Customer DataFrame: {df_customer_pl.shape[0]}")
print(df_customer_pl.head(5))
print(df_customer_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Customer DataFrame: 100
```

```txt
shape: (5, 5)
┌─────────────┬───────────────┬─────────────┬───────┬─────────────┐
│ customer_id ┆ customer_name ┆ city        ┆ state ┆ segment     │
│ ---         ┆ ---           ┆ ---         ┆ ---   ┆ ---         │
│ i64         ┆ str           ┆ str         ┆ str   ┆ str         │
╞═════════════╪═══════════════╪═════════════╪═══════╪═════════════╡
│ 1           ┆ Customer 1    ┆ Phoenix     ┆ NY    ┆ Corporate   │
│ 2           ┆ Customer 2    ┆ Phoenix     ┆ CA    ┆ Home Office │
│ 3           ┆ Customer 3    ┆ Phoenix     ┆ NY    ┆ Home Office │
│ 4           ┆ Customer 4    ┆ Los Angeles ┆ NY    ┆ Consumer    │
│ 5           ┆ Customer 5    ┆ Los Angeles ┆ IL    ┆ Home Office │
└─────────────┴───────────────┴─────────────┴───────┴─────────────┘
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

### Pandas

In Pandas, we can use boolean indexing to filter rows based on specific conditions. As you can see in this first example, this looks like using square brackets, within which we define a column and a condition. In the below example, we can use string values to filter categorical data.

For more information about filtering in Pandas, see the [Pandas documentation on filtering][pandas-subsetting].

```python {.pandas linenums="1" title="Filter sales data for specific category"}
electronics_sales_pd: pd.DataFrame = df_sales_pd[df_sales_pd["category"] == "Electronics"]
print(f"Number of Electronics Sales: {len(electronics_sales_pd)}")
print(electronics_sales_pd.head(5))
print(electronics_sales_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Number of Electronics Sales: 28
```

```txt
         date  customer_id  product_id     category  sales_amount  quantity
1  2023-01-02           93          41  Electronics        453.94         5
3  2023-01-04           72          15  Electronics        184.17         7
8  2023-01-09           75           9  Electronics        746.73         2
10 2023-01-11           88           1  Electronics        314.98         9
11 2023-01-12           24          44  Electronics        547.11         8
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
|    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
|    8 | 2023-01-09 00:00:00 |          75 |          9 | Electronics |       746.73 |        2 |
|   10 | 2023-01-11 00:00:00 |          88 |          1 | Electronics |       314.98 |        9 |
|   11 | 2023-01-12 00:00:00 |          24 |         44 | Electronics |       547.11 |        8 |

</div>

### SQL

In SQL, we can use the `WHERE` clause to filter rows based on specific conditions. The syntax should be very familiar to anyone who has worked with SQL before. We can use the [`pd.read_sql()`][pandas-read_sql] function to execute SQL queries and retrieve the data from the database. The result is a Pandas DataFrame that contains only the rows that match the specified condition. In the below example, we filter for sales in the "Electronics" category.

For more information about filtering in SQL, see the [SQL WHERE clause documentation][sqlite-clause].

```python {.sql linenums="1" title="Filter sales for a specific category"}
electronics_sales_txt: str = """
    SELECT *
    FROM sales
    WHERE category = 'Electronics'
"""
electronics_sales_sql: pd.DataFrame = pd.read_sql(electronics_sales_txt, conn)
print(f"Number of Electronics Sales: {len(electronics_sales_sql)}")
print(pd.read_sql(electronics_sales_txt + "LIMIT 5", conn))
print(pd.read_sql(electronics_sales_txt + "LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt
Number of Electronics Sales: 28
```

```txt
                  date  customer_id  product_id     category  sales_amount  quantity
0  2023-01-02 00:00:00           93          41  Electronics        453.94         5
1  2023-01-04 00:00:00           72          15  Electronics        184.17         7
2  2023-01-09 00:00:00           75           9  Electronics        746.73         2
3  2023-01-11 00:00:00           88           1  Electronics        314.98         9
4  2023-01-12 00:00:00           24          44  Electronics        547.11         8
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
|    1 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
|    2 | 2023-01-09 00:00:00 |          75 |          9 | Electronics |       746.73 |        2 |
|    3 | 2023-01-11 00:00:00 |          88 |          1 | Electronics |       314.98 |        9 |
|    4 | 2023-01-12 00:00:00 |          24 |         44 | Electronics |       547.11 |        8 |

</div>

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

### Polars

In Polars, we can use the [`.filter()`][polars-filter] method to filter rows based on specific conditions. The syntax is similar to Pandas, where we can specify the condition using column expressions. In the below example, we filter for sales in the "Electronics" category.

For more information about filtering in Polars, see the [Polars documentation on filtering][polars-filtering].

```python {.polars linenums="1" title="Filter sales for a specific category"}
electronics_sales_pl: pl.DataFrame = df_sales_pl.filter(df_sales_pl["category"] == "Electronics")
print(f"Number of Electronics Sales: {len(electronics_sales_pl)}")
print(electronics_sales_pl.head(5))
print(electronics_sales_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Number of Electronics Sales: 28
```

```txt
shape: (5, 6)
┌─────────────────────┬─────────────┬────────────┬─────────────┬──────────────┬──────────┐
│ date                ┆ customer_id ┆ product_id ┆ category    ┆ sales_amount ┆ quantity │
│ ---                 ┆ ---         ┆ ---        ┆ ---         ┆ ---          ┆ ---      │
│ datetime[ns]        ┆ i64         ┆ i64        ┆ str         ┆ f64          ┆ i64      │
╞═════════════════════╪═════════════╪════════════╪═════════════╪══════════════╪══════════╡
│ 2023-01-02 00:00:00 ┆ 93          ┆ 41         ┆ Electronics ┆ 453.94       ┆ 5        │
│ 2023-01-04 00:00:00 ┆ 72          ┆ 15         ┆ Electronics ┆ 184.17       ┆ 7        │
│ 2023-01-09 00:00:00 ┆ 75          ┆ 9          ┆ Electronics ┆ 746.73       ┆ 2        │
│ 2023-01-11 00:00:00 ┆ 88          ┆ 1          ┆ Electronics ┆ 314.98       ┆ 9        │
│ 2023-01-12 00:00:00 ┆ 24          ┆ 44         ┆ Electronics ┆ 547.11       ┆ 8        │
└─────────────────────┴─────────────┴────────────┴─────────────┴──────────────┴──────────┘
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

### Pandas

When it comes to numerical filtering in Pandas, the process is similar to the previous example, where we use boolean indexing to filter rows based on a given condition condition, but here we use a numerical value instead of a string value. In the below example, we filter for sales amounts greater than `500`.

```python {.pandas linenums="1" title="Filter for high value transactions"}
high_value_sales_pd: pd.DataFrame = df_sales_pd[df_sales_pd["sales_amount"] > 500]
print(f"Number of high-value Sales: {len(high_value_sales_pd)}")
print(high_value_sales_pd.head(5))
print(high_value_sales_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Number of high-value Sales: 43
```

```txt
         date  customer_id  product_id     category  sales_amount  quantity
2  2023-01-03           15          29         Home        994.51         5
8  2023-01-09           75           9  Electronics        746.73         2
9  2023-01-10           75          24        Books        723.73         6
11 2023-01-12           24          44  Electronics        547.11         8
12 2023-01-13            3           8     Clothing        513.73         5
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
|    8 | 2023-01-09 00:00:00 |          75 |          9 | Electronics |       746.73 |        2 |
|    9 | 2023-01-10 00:00:00 |          75 |         24 | Books       |       723.73 |        6 |
|   11 | 2023-01-12 00:00:00 |          24 |         44 | Electronics |       547.11 |        8 |
|   12 | 2023-01-13 00:00:00 |           3 |          8 | Clothing    |       513.73 |        5 |

</div>

### SQL

When it comes to numerical filtering in SQL, the process is similar to the previous example, where we use the `WHERE` clause to filter rows based on a given condition, but here we use a numerical value instead of a string value. In the below example, we filter for sales amounts greater than `500`.

```python {.sql linenums="1" title="Filter for high value transactions"}
high_value_sales_txt: str = """
    SELECT *
    FROM sales
    WHERE sales_amount > 500
"""
high_value_sales_sql: pd.DataFrame = pd.read_sql(high_value_sales_txt, conn)
print(f"Number of high-value Sales: {len(high_value_sales_sql)}")
print(pd.read_sql(high_value_sales_txt + "LIMIT 5", conn))
print(pd.read_sql(high_value_sales_txt + "LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt
Number of high-value Sales: 43
```

```txt
                  date  customer_id  product_id     category  sales_amount  quantity
0  2023-01-03 00:00:00           15          29         Home        994.51         5
1  2023-01-09 00:00:00           75           9  Electronics        746.73         2
2  2023-01-10 00:00:00           75          24        Books        723.73         6
3  2023-01-12 00:00:00           24          44  Electronics        547.11         8
4  2023-01-13 00:00:00            3           8     Clothing        513.73         5
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
|    1 | 2023-01-09 00:00:00 |          75 |          9 | Electronics |       746.73 |        2 |
|    2 | 2023-01-10 00:00:00 |          75 |         24 | Books       |       723.73 |        6 |
|    3 | 2023-01-12 00:00:00 |          24 |         44 | Electronics |       547.11 |        8 |
|    4 | 2023-01-13 00:00:00 |           3 |          8 | Clothing    |       513.73 |        5 |

</div>

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

### Polars

When it comes to numerical filtering in Polars, the process is similar to the previous example, where we use the [`.filter()`][polars-filter] method to filter rows based on a given condition, but here we use a numerical value instead of a string value. In the below example, we filter for sales amounts greater than `500`.

Also note here that we have used the [`pl.col()`][polars-col] function to specify the column we want to filter on. This is different from the previous examples, where we used the column name directly. The use of [`pl.col()`][polars-col] is a common practice in Polars to specify the column name in a more readable way.

```python {.polars linenums="1" title="Filter for high value transactions"}
high_value_sales_pl: pl.DataFrame = df_sales_pl.filter(pl.col("sales_amount") > 500)
print(f"Number of high-value Sales: {len(high_value_sales_pl)}")
print(high_value_sales_pl.head(5))
print(high_value_sales_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Number of high-value Sales: 43
```

```txt
shape: (5, 6)
┌─────────────────────┬─────────────┬────────────┬─────────────┬──────────────┬──────────┐
│ date                ┆ customer_id ┆ product_id ┆ category    ┆ sales_amount ┆ quantity │
│ ---                 ┆ ---         ┆ ---        ┆ ---         ┆ ---          ┆ ---      │
│ datetime[ns]        ┆ i64         ┆ i64        ┆ str         ┆ f64          ┆ i64      │
╞═════════════════════╪═════════════╪════════════╪═════════════╪══════════════╪══════════╡
│ 2023-01-03 00:00:00 ┆ 15          ┆ 29         ┆ Home        ┆ 994.51       ┆ 5        │
│ 2023-01-09 00:00:00 ┆ 75          ┆ 9          ┆ Electronics ┆ 746.73       ┆ 2        │
│ 2023-01-10 00:00:00 ┆ 75          ┆ 24         ┆ Books       ┆ 723.73       ┆ 6        │
│ 2023-01-12 00:00:00 ┆ 24          ┆ 44         ┆ Electronics ┆ 547.11       ┆ 8        │
│ 2023-01-13 00:00:00 ┆ 3           ┆ 8          ┆ Clothing    ┆ 513.73       ┆ 5        │
└─────────────────────┴─────────────┴────────────┴─────────────┴──────────────┴──────────┘
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


### Pandas

To select specific columns in Pandas, we can use the double square brackets syntax to specify the columns we want to keep in the DataFrame. This allows us to create a new DataFrame with only the relevant columns.

For more information about selecting specific columns, see the [Pandas documentation on selecting columns][pandas-subsetting].

```python {.pandas linenums="1" title="Select specific columns"}
sales_summary_pd: pd.DataFrame = df_sales_pd[["date", "category", "sales_amount"]]
print(f"Sales Summary DataFrame: {len(sales_summary_pd)}")
print(sales_summary_pd.head(5))
print(sales_summary_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Sales Summary DataFrame: 100
```

```txt
        date     category  sales_amount
0 2023-01-01         Food        490.76
1 2023-01-02  Electronics        453.94
2 2023-01-03         Home        994.51
3 2023-01-04  Electronics        184.17
4 2023-01-05         Food         27.89
```

|      | date                | category    | sales_amount |
| ---: | :------------------ | :---------- | -----------: |
|    0 | 2023-01-01 00:00:00 | Food        |       490.76 |
|    1 | 2023-01-02 00:00:00 | Electronics |       453.94 |
|    2 | 2023-01-03 00:00:00 | Home        |       994.51 |
|    3 | 2023-01-04 00:00:00 | Electronics |       184.17 |
|    4 | 2023-01-05 00:00:00 | Food        |        27.89 |

</div>

### SQL

To select specific columns in SQL, we can use the `SELECT` statement to specify the columns we want to retrieve from the table. This allows us to create a new DataFrame with only the relevant columns. We can use the [`pd.read_sql()`][pandas-read_sql] function to execute SQL queries and retrieve the data from the database.

For more information about selecting specific columns in SQL, see the [SQL SELECT statement documentation][sqlite-select].

```python {.sql linenums="1" title="Select specific columns"}
sales_summary_txt: str = """
    SELECT date, category, sales_amount
    FROM sales
"""
sales_summary_sql: pd.DataFrame = pd.read_sql(sales_summary_txt, conn)
print(f"Selected columns in Sales: {len(sales_summary_sql)}")
print(pd.read_sql(sales_summary_txt + "LIMIT 5", conn))
print(pd.read_sql(sales_summary_txt + "LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt
Selected columns in Sales: 100
```

```txt
                  date     category  sales_amount
0  2023-01-01 00:00:00         Food        490.76
1  2023-01-02 00:00:00  Electronics        453.94
2  2023-01-03 00:00:00         Home        994.51
3  2023-01-04 00:00:00  Electronics        184.17
4  2023-01-05 00:00:00         Food         27.89
```

|      | date                | category    | sales_amount |
| ---: | :------------------ | :---------- | -----------: |
|    0 | 2023-01-01 00:00:00 | Food        |       490.76 |
|    1 | 2023-01-02 00:00:00 | Electronics |       453.94 |
|    2 | 2023-01-03 00:00:00 | Home        |       994.51 |
|    3 | 2023-01-04 00:00:00 | Electronics |       184.17 |
|    4 | 2023-01-05 00:00:00 | Food        |        27.89 |

</div>

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

### Polars

To select specific columns in Polars, we can use the [`.select()`][polars-select] method to specify the columns we want to keep in the DataFrame. This allows us to create a new DataFrame with only the relevant columns.

```python {.polars linenums="1" title="Select specific columns"}
sales_summary_pl: pl.DataFrame = df_sales_pl.select(["date", "category", "sales_amount"])
print(f"Sales Summary DataFrame: {len(sales_summary_pl)}")
print(sales_summary_pl.head(5))
print(sales_summary_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales Summary DataFrame: 100
```

```txt
shape: (5, 3)
┌─────────────────────┬─────────────┬──────────────┐
│ date                ┆ category    ┆ sales_amount │
│ ---                 ┆ ---         ┆ ---          │
│ datetime[ns]        ┆ str         ┆ f64          │
╞═════════════════════╪═════════════╪══════════════╡
│ 2023-01-01 00:00:00 ┆ Food        ┆ 490.76       │
│ 2023-01-02 00:00:00 ┆ Electronics ┆ 453.94       │
│ 2023-01-03 00:00:00 ┆ Home        ┆ 994.51       │
│ 2023-01-04 00:00:00 ┆ Electronics ┆ 184.17       │
│ 2023-01-05 00:00:00 ┆ Food        ┆ 27.89        │
└─────────────────────┴─────────────┴──────────────┘
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

### Pandas

In Pandas, we can use the [`.agg()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html) method to perform aggregation operations on DataFrames. This method allows us to apply multiple aggregation functions to different columns in a single operation.

```python {.pandas linenums="1" title="Basic aggregation"}
sales_stats: pd.DataFrame = df_sales_pd.agg(
    {
        "sales_amount": ["sum", "mean", "min", "max", "count"],
        "quantity": ["sum", "mean", "min", "max"],
    }
)
print(f"Sales Statistics: {len(sales_stats)}")
print(sales_stats)
print(sales_stats.to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="Basic aggregation"}
sales_stats_sql: str = """
    SELECT
        SUM(sales_amount) AS sales_sum,
        AVG(sales_amount) AS sales_mean,
        MIN(sales_amount) AS sales_min,
        MAX(sales_amount) AS sales_max,
        COUNT(*) AS sales_count,
        SUM(quantity) AS quantity_sum,
        AVG(quantity) AS quantity_mean,
        MIN(quantity) AS quantity_min,
        MAX(quantity) AS quantity_max
    FROM sales
"""
print(f"Sales Statistics: {len(pd.read_sql(sales_stats_sql, conn))}")
print(pd.read_sql(sales_stats_sql, conn))
print(pd.read_sql(sales_stats_sql, conn).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="Basic aggregation"}
sales_stats: psDataFrame = df_sales_ps.agg(
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
print(f"Sales Statistics: {sales_stats.count()}")
sales_stats.show(5)
print(sales_stats.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="Basic aggregation"}
sales_stats: pl.DataFrame = df_sales_pl.select(
    pl.col("sales_amount").sum().alias("sales_sum"),
    pl.col("sales_amount").mean().alias("sales_mean"),
    pl.col("sales_amount").min().alias("sales_min"),
    pl.col("sales_amount").max().alias("sales_max"),
    pl.col("quantity").sum().alias("quantity_sum"),
    pl.col("quantity").mean().alias("quantity_mean"),
    pl.col("quantity").min().alias("quantity_min"),
    pl.col("quantity").max().alias("quantity_max"),
)
print(f"Sales Statistics: {len(sales_stats)}")
print(sales_stats)
print(sales_stats.to_pandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

It is also possible to group data by a specific column and then apply aggregation functions to summarize the data.

### Pandas

This is done using the [`.groupby()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) method to group data by one or more columns and then apply aggregation functions to summarize the data, followed by the [`.agg()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html) method.

```python {.pandas linenums="1" title="Group by category and aggregate"}
category_sales: pd.DataFrame = df_sales_pd.groupby("category").agg(
    {
        "sales_amount": ["sum", "mean", "count"],
        "quantity": "sum",
    }
)
print(f"Category Sales Summary: {len(category_sales)}")
print(category_sales)
print(category_sales.to_pandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="Group by category and aggregate"}
category_sales_sql: str = """
    SELECT
        category,
        SUM(sales_amount) AS total_sales,
        AVG(sales_amount) AS average_sales,
        COUNT(*) AS transaction_count,
        SUM(quantity) AS total_quantity
    FROM sales
    GROUP BY category
"""
print(f"Category Sales Summary: {len(pd.read_sql(category_sales_sql, conn))}")
print(pd.read_sql(category_sales_sql + "LIMIT 5", conn))
print(pd.read_sql(category_sales_sql + "LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="Group by category and aggregate"}
category_sales: psDataFrame = df_sales_ps.groupBy("category").agg(
    F.sum("sales_amount").alias("total_sales"),
    F.avg("sales_amount").alias("average_sales"),
    F.count("*").alias("transaction_count"),
    F.sum("quantity").alias("total_quantity"),
)
print(f"Category Sales Summary: {category_sales.count()}")
category_sales.show(5)
print(category_sales.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="Group by category and aggregate"}
category_sales: pl.DataFrame = df_sales_pl.group_by("category").agg(
    pl.col("sales_amount").sum().alias("total_sales"),
    pl.col("sales_amount").mean().alias("average_sales"),
    pl.col("sales_amount").count().alias("transaction_count"),
    pl.col("quantity").sum().alias("total_quantity"),
)
print(f"Category Sales Summary: {len(category_sales)}")
print(category_sales.head(5))
print(category_sales.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

We can rename the columns for clarity by simply assigning new names.

### Pandas

In Pandas, we use the [`.columns`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html) attribute of the DataFrame. This makes it easier to understand the results of the aggregation.

```python {.pandas linenums="1" title="Rename columns for clarity"}
category_sales.columns = [
    "total_sales",
    "average_sales",
    "transaction_count",
    "total_quantity",
]
print(f"Renamed Category Sales Summary: {len(category_sales)}")
print(category_sales.head(5))
print(category_sales.head(5).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="Rename columns for clarity"}
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="Rename columns for clarity"}
category_sales: psDataFrame = category_sales.withColumnsRenamed(
    {
        "total_sales": "Total Sales",
        "average_sales": "Average Sales",
        "transaction_count": "Transaction Count",
        "total_quantity": "Total Quantity",
    }
)
print(f"Renamed Category Sales Summary: {category_sales.count()}")
category_sales.show(5)
print(category_sales.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="Rename columns for clarity"}
category_sales: pl.DataFrame = category_sales.rename(
    {
        "total_sales": "Total Sales",
        "average_sales": "Average Sales",
        "transaction_count": "Transaction Count",
        "total_quantity": "Total Quantity",
    }
)
print(f"Renamed Category Sales Summary: {len(category_sales)}")
print(category_sales.head(5))
print(category_sales.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

Having aggregated the data, we can now visualize the results using [Plotly](https://plotly.com/python/). This allows us to create interactive visualizations that can help us better understand the data.

### Pandas

```python {.pandas linenums="1" title="Plot the results"}
fig: go.Figure = px.bar(
    category_sales.reset_index(),
    x="category",
    y="total_sales",
    title="Total Sales by Category",
    text="transaction_count",
    labels={"total_sales": "Total Sales ($)", "category": "Product Category"},
)
fig.show()
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="Plot the results"}
fig: go.Figure = px.bar(
    pd.read_sql(category_sales_sql, conn),
    x="category",
    y="total_sales",
    title="Total Sales by Category",
    text="transaction_count",
    labels={"total_sales": "Total Sales ($)", "category": "Product Category"},
)
fig.show()
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="Plot the results"}
category_sales_pd: pd.DataFrame = category_sales.toPandas()
fig: go.Figure = px.bar(
    category_sales_pd,
    x="category",
    y="Total Sales",
    title="Total Sales by Category",
    text="Transaction Count",
    labels={"Total Sales": "Total Sales ($)", "category": "Product Category"},
)
fig.show()
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="Plot the results"}
fig: go.Figure = px.bar(
    category_sales,
    x="category",
    y="Total Sales",
    title="Total Sales by Category",
    text="Transaction Count",
    labels={"Total Sales": "Total Sales ($)", "category": "Product Category"},
)
fig.show()
```

<div class="result" markdown>

```txt

```

</div>

## 3. Joining

The third section will demonstrate how to join DataFrames to combine data from different sources. This is a common operation in data analysis, allowing us to enrich our data with additional information.

Here, we will join the `sales` DataFrame with the `product` DataFrame to get additional information about the products sold.

### Pandas

In Pandas, we can use the [`pd.merge()`](https://pandas.pydata.org/docs/reference/api/pandas.merge.html) method to join DataFrames. This method allows us to specify the columns to join on and the type of join (inner, outer, left, or right).

```python {.pandas linenums="1" title="Join sales with product data"}
sales_with_product: pd.DataFrame = pd.merge(
    left=df_sales_pd,
    right=df_product_pd[["product_id", "product_name", "price"]],
    on="product_id",
    how="left",
)
print(f"Sales with Product Information: {len(sales_with_product)}")
print(sales_with_product.head(5))
print(sales_with_product.head(5).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="Join sales with product data"}
sales_with_product_sql: str = """
    SELECT s.*, p.product_name, p.price
    FROM sales s
    LEFT JOIN product p ON s.product_id = p.product_id
"""
print(f"Sales with Product Data: {len(pd.read_sql(sales_with_product_sql, conn))}")
print(pd.read_sql(sales_with_product_sql + "LIMIT 5", conn))
print(pd.read_sql(sales_with_product_sql + "LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="Join sales with product data"}
sales_with_product: psDataFrame = df_sales_ps.join(
    other=df_product_ps.select("product_id", "product_name", "price"),
    on="product_id",
    how="left",
)
print(f"Sales with Product Information: {sales_with_product.count()}")
sales_with_product.show(5)
print(sales_with_product.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="Join sales with product data"}
sales_with_product: pl.DataFrame = df_sales_pl.join(
    df_product_pl.select(["product_id", "product_name", "price"]),
    on="product_id",
    how="left",
)
print(f"Sales with Product Information: {len(sales_with_product)}")
print(sales_with_product.head(5))
print(sales_with_product.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

In the next step, we will join the resulting DataFrame with the `customer` DataFrame to get customer information for each sale. This allows us to create a complete view of the sales data, including product and customer details.

### Pandas

```python {.pandas linenums="1" title="Join with customer information to get a complete view"}
complete_sales: pd.DataFrame = pd.merge(
    sales_with_product,
    df_customer_pd[["customer_id", "customer_name", "city", "state"]],
    on="customer_id",
    how="left",
)
print(f"Complete Sales Data with Customer Information: {len(complete_sales)}")
print(complete_sales.head(5))
print(complete_sales.head(5).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="Join with customer information to get a complete view"}
complete_sales_sql: str = """
    SELECT
        s.*,
        p.product_name,
        p.price,
        c.customer_name,
        c.city,
        c.state
    FROM sales s
    LEFT JOIN product p ON s.product_id = p.product_id
    LEFT JOIN customer c ON s.customer_id = c.customer_id
"""
print(f"Complete Sales Data: {len(pd.read_sql(complete_sales_sql, conn))}")
print(pd.read_sql(complete_sales_sql + "LIMIT 5", conn))
print(pd.read_sql(complete_sales_sql + "LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="Join with customer information to get a complete view"}
complete_sales: psDataFrame = sales_with_product.alias("s").join(
    other=df_customer_ps.select("customer_id", "customer_name", "city", "state").alias("c"),
    on="customer_id",
    how="left",
)
print(f"Complete Sales Data with Customer Information: {complete_sales.count()}")
complete_sales.show(5)
print(complete_sales.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="Join with customer information to get a complete view"}
complete_sales: pl.DataFrame = sales_with_product.join(
    df_customer_pl.select(["customer_id", "customer_name", "city", "state"]),
    on="customer_id",
    how="left",
)
print(f"Complete Sales Data with Customer Information: {len(complete_sales)}")
print(complete_sales.head(5))
print(complete_sales.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

Once we have the complete sales data, we can calculate the revenue for each sale by multiplying the price and quantity (columns from different tables). We can also compare this calculated revenue with the sales amount to identify any discrepancies.

### Pandas

```python {.pandas linenums="1" title="Calculate revenue and compare with sales amount"}
complete_sales["calculated_revenue"] = complete_sales["price"] * complete_sales["quantity"]
complete_sales["price_difference"] = complete_sales["sales_amount"] - complete_sales["calculated_revenue"]
print(f"Complete Sales Data with Calculated Revenue and Price Difference: {len(complete_sales)}")
print(complete_sales[["sales_amount", "price", "quantity", "calculated_revenue", "price_difference"]].head(5))
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="Calculate revenue and compare with sales amount"}
revenue_comparison_sql: str = """
    SELECT
        s.sales_amount,
        p.price,
        s.quantity,
        (p.price * s.quantity) AS calculated_revenue,
        (s.sales_amount - (p.price * s.quantity)) AS price_difference
    FROM sales s
    LEFT JOIN product p ON s.product_id = p.product_id
"""
print(f"Revenue Comparison: {len(pd.read_sql(revenue_comparison_sql, conn))}")
print(pd.read_sql(revenue_comparison_sql + "LIMIT 5", conn))
print(pd.read_sql(revenue_comparison_sql + "LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="Calculate revenue and compare with sales amount"}
complete_sales: psDataFrame = complete_sales.withColumns(
    {
        "calculated_revenue": complete_sales["price"] * complete_sales["quantity"],
        "price_difference": F.expr("sales_amount - (price * quantity)"),
    },
)
print(f"Complete Sales Data with Calculated Revenue and Price Difference: {complete_sales.count()}")
complete_sales.select("sales_amount", "price", "quantity", "calculated_revenue", "price_difference").show(5)
print(
    complete_sales.select("sales_amount", "price", "quantity", "calculated_revenue", "price_difference")
    .limit(5)
    .toPandas()
    .to_markdown()
)
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="Calculate revenue and compare with sales amount"}
complete_sales: pl.DataFrame = complete_sales.with_columns(
    (pl.col("price") * pl.col("quantity")).alias("calculated_revenue"),
    (pl.col("sales_amount") - (pl.col("price") * pl.col("quantity"))).alias("price_difference"),
)
print(f"Complete Sales Data with Calculated Revenue and Price Difference: {len(complete_sales)}")
print(complete_sales.select(["sales_amount", "price", "quantity", "calculated_revenue", "price_difference"]).head(5))
```

<div class="result" markdown>

```txt

```

</div>

## 4. Window Functions

Window functions are a powerful feature in Pandas that allow us to perform calculations across a set of rows related to the current row. This is particularly useful for time series data, where we may want to calculate rolling averages, cumulative sums, or other metrics based on previous or subsequent rows.

To understand more about the nuances of the window functions, check out some of these guides:

- [Analyzing data with window functions](https://docs.snowflake.com/en/user-guide/functions-window-using)
- [SQL Window Functions Visualized](https://medium.com/learning-sql/sql-window-function-visualized-fff1927f00f2)

In this section, we will demonstrate how to use window functions to analyze sales data over time. We will start by converting the `date` column to a datetime type, which is necessary for time-based calculations. We will then group the data by date and calculate the total sales for each day.

### Pandas

```python {.pandas linenums="1" title="Time-based window function"}
df_sales_pd["date"] = pd.to_datetime(df_sales_pd["date"])  # Ensure correct date type
daily_sales: pd.DataFrame = (
    df_sales_pd.groupby(df_sales_pd["date"].dt.date)["sales_amount"].sum().reset_index().sort_values("date")
)
print(f"Daily Sales Summary: {len(daily_sales)}")
print(daily_sales.head(5))
print(daily_sales.head(5).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="Time-based window function"}
daily_sales_sql: str = """
    SELECT
        date,
        SUM(sales_amount) AS total_sales
    FROM sales
    GROUP BY date
    ORDER BY date
"""
print(f"Daily Sales Data: {len(pd.read_sql(daily_sales_sql, conn))}")
print(pd.read_sql(daily_sales_sql + "LIMIT 5", conn))
print(pd.read_sql(daily_sales_sql + "LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="Time-based window function"}
df_sales_ps: psDataFrame = df_sales_ps.withColumn("date", F.to_date(df_sales_ps["date"]))
daily_sales: psDataFrame = (
    df_sales_ps.groupBy("date")
    .agg(
        F.sum("sales_amount").alias("total_sales"),
    )
    .orderBy("date")
)
print(f"Daily Sales Summary: {daily_sales.count()}")
daily_sales.show(5)
print(daily_sales.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="Time-based window function"}
df_sales_pl: pl.DataFrame = df_sales_pl.with_columns(pl.col("date").cast(pl.Date))
```

<div class="result" markdown>

```txt

```

</div>

```python {.polars linenums="1" title="TITLE"}
daily_sales: pl.DataFrame = (
    df_sales_pl.group_by("date")
    .agg(
        pl.col("sales_amount").sum().alias("total_sales"),
    )
    .sort("date")
)
print(f"Daily Sales Summary: {len(daily_sales)}")
print(daily_sales.head(5))
print(daily_sales.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

Next, we will calculate the rolling average of sales over a 7-day window.

### Pandas

This is done using the [`.rolling()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html) method, which allows us to specify the window size and the minimum number of periods required for the calculation.

```python {.pandas linenums="1" title="TITLE"}
# Calculate rolling averages (7-day moving average)
daily_sales["7d_moving_avg"] = daily_sales["sales_amount"].rolling(window=7, min_periods=1).mean()
print(f"Daily Sales with 7-Day Moving Average: {len(daily_sales)}")
print(daily_sales.head(5))
print(daily_sales.head(5).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="TITLE"}
# Window functions for lead and lag
window_sql: str = """
    SELECT
        date AS sale_date,
        SUM(sales_amount) AS sales_amount,
        AVG(sales_amount) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_avg,
        LAG(SUM(sales_amount)) OVER (ORDER BY date) AS previous_day_sales,
        LEAD(SUM(sales_amount)) OVER (ORDER BY date) AS next_day_sales,
        SUM(sales_amount) - LAG(SUM(sales_amount)) OVER (ORDER BY date) AS day_over_day_change
    FROM sales
    GROUP BY date
    ORDER BY date
"""
window_df: pd.DataFrame = pd.read_sql(window_sql, conn)
print(f"Window Functions: {len(window_df)}")
print(pd.read_sql(window_sql + "LIMIT 5", conn))
print(pd.read_sql(window_sql + "LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="TITLE"}
# Calculate day-over-day change
daily_sales: psDataFrame = daily_sales.withColumns(
    {
        "day_over_day_change": F.expr("total_sales - previous_day_sales"),
        "pct_change": (F.expr("total_sales / previous_day_sales - 1") * 100).alias("pct_change"),
    }
)
print(f"Daily Sales with Day-over-Day Change: {daily_sales.count()}")
daily_sales.show(5)
print(daily_sales.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="TITLE"}
# Calculate day-over-day change
daily_sales: pl.DataFrame = daily_sales.with_columns(
    (pl.col("total_sales") - pl.col("previous_day_sales")).alias("day_over_day_change"),
    (pl.col("total_sales") / pl.col("previous_day_sales") - 1).alias("pct_change") * 100,
)
print(f"Daily Sales with Day-over-Day Change: {len(daily_sales)}")
print(daily_sales.head(5))
print(daily_sales.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

Next, we will calculate the lag and lead values for the sales amount. This allows us to compare the current day's sales with the previous and next days' sales.

### Pandas

This is done using the [`.shift()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html) method, which shifts the values in a column by a specified number of periods. Note that the [`.shift()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html) method simply shifts the values in the column, so we can use it to create lag and lead columns. This function itself does not need to be ordered because it assumes that the DataFrame is already ordered; if you want it to be ordered, you can use the [`.sort_values()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html) method before applying [`.shift()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html).

```python {.pandas linenums="1" title="TITLE"}
# Calculate lag and lead
daily_sales["previous_day_sales"] = daily_sales["sales_amount"].shift(1)
daily_sales["next_day_sales"] = daily_sales["sales_amount"].shift(-1)
print(f"Daily Sales with Lag and Lead: {len(daily_sales)}")
print(daily_sales.head(5))
print(daily_sales.head(5).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="TITLE"}
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="TITLE"}
# Define a window specification for lead/lag functions
window_spec: psDataFrame = Window.orderBy("date")

# Calculate lead and lag
daily_sales: psDataFrame = daily_sales.withColumns(
    {
        "previous_day_sales": F.lag("total_sales").over(window_spec),
        "next_day_sales": F.expr("LEAD(total_sales) OVER (ORDER BY date)"),
    },
)
print(f"Daily Sales with Lead and Lag: {daily_sales.count()}")
daily_sales.show(5)
print(daily_sales.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="TITLE"}
# Calculate lead and lag
daily_sales: pl.DataFrame = daily_sales.with_columns(
    pl.col("total_sales").shift(1).alias("previous_day_sales"),
    pl.col("total_sales").shift(-1).alias("next_day_sales"),
)
print(f"Daily Sales with Lead and Lag: {len(daily_sales)}")
print(daily_sales.head(5))
print(daily_sales.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

Now, we can calculate the day-over-day change in sales. This is done by subtracting the previous day's sales from the current day's sales.

### Pandas

We can also calculate the percentage change in sales using the [`.pct_change()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html) method, which calculates the percentage change between the current and previous values.

```python {.pandas linenums="1" title="TITLE"}
# Calculate day-over-day change
daily_sales["day_over_day_change"] = daily_sales["sales_amount"].pct_change() - daily_sales["previous_day_sales"]
daily_sales["pct_change"] = daily_sales["sales_amount"].pct_change() * 100
print(f"Daily Sales with Day-over-Day Change: {len(daily_sales)}")
print(daily_sales.head(5))
print(daily_sales.head(5).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="TITLE"}
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="TITLE"}
# Calculate 7-day moving average
daily_sales: psDataFrame = daily_sales.withColumns(
    {
        "7d_moving_avg": F.avg("total_sales").over(Window.orderBy("date").rowsBetween(-6, 0)),
        "7d_rolling_avg": F.expr("AVG(total_sales) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)"),
    }
)
print(f"Daily Sales with 7-Day Moving Average: {daily_sales.count()}")
daily_sales.show(5)
print(daily_sales.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="TITLE"}
# Calculate 7-day moving average
daily_sales: pl.DataFrame = daily_sales.with_columns(
    pl.col("total_sales").rolling_mean(window_size=7, min_periods=1).alias("7d_moving_avg"),
)
print(f"Daily Sales with 7-Day Moving Average: {len(daily_sales)}")
print(daily_sales.head(5))
print(daily_sales.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

Finally, we can visualize the daily sales data along with the 7-day moving average using Plotly. This allows us to see the trends in sales over time and how the moving average smooths out the fluctuations in daily sales.

### Pandas

```python {.pandas linenums="1" title="Plot results"}
fig: go.Figure = (
    go.Figure()
    .add_trace(
        go.Scatter(
            x=daily_sales["date"],
            y=daily_sales["sales_amount"],
            mode="lines",
            name="Daily Sales",
        )
    )
    .add_trace(
        go.Scatter(
            x=daily_sales["date"],
            y=daily_sales["7d_moving_avg"],
            mode="lines",
            name="7-Day Moving Average",
            line=dict(width=3),
        ),
    )
    .update_layout(
        title="Daily Sales with 7-Day Moving Average",
        xaxis_title="Date",
        yaxis_title="Sales Amount ($)",
    )
)
fig.show()
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="Plot results"}
fig: go.Figure = (
    go.Figure()
    .add_trace(
        go.Scatter(
            x=window_df["sale_date"],
            y=window_df["sales_amount"],
            mode="lines",
            name="Daily Sales",
        )
    )
    .add_trace(
        go.Scatter(
            x=window_df["sale_date"],
            y=window_df["rolling_7d_avg"],
            mode="lines",
            name="7-Day Moving Average",
            line=dict(width=3),
        )
    )
    .update_layout(
        title="Daily Sales with 7-Day Moving Average",
        xaxis_title="Date",
        yaxis_title="Sales Amount ($)",
    )
)
fig.show()
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="Plot results"}
fig: go.Figure = (
    go.Figure()
    .add_trace(
        go.Scatter(
            x=daily_sales.toPandas()["date"],
            y=daily_sales.toPandas()["total_sales"],
            mode="lines",
            name="Daily Sales",
        )
    )
    .add_trace(
        go.Scatter(
            x=daily_sales.toPandas()["date"],
            y=daily_sales.toPandas()["7d_moving_avg"],
            mode="lines",
            name="7-Day Moving Average",
            line=dict(width=3),
        ),
    )
    .update_layout(
        title="Daily Sales with 7-Day Moving Average",
        xaxis_title="Date",
        yaxis_title="Sales Amount ($)",
    )
)
fig.show()
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="Plot results"}
fig: go.Figure = (
    go.Figure()
    .add_trace(
        go.Scatter(
            x=daily_sales["date"].to_list(),
            y=daily_sales["total_sales"].to_list(),
            mode="lines",
            name="Daily Sales",
        )
    )
    .add_trace(
        go.Scatter(
            x=daily_sales["date"].to_list(),
            y=daily_sales["7d_moving_avg"].to_list(),
            mode="lines",
            name="7-Day Moving Average",
            line=dict(width=3),
        )
    )
    .update_layout(
        title="Daily Sales with 7-Day Moving Average",
        xaxis_title="Date",
        yaxis_title="Sales Amount ($)",
    )
)
fig.show()
```

<div class="result" markdown>

```txt

```

</div>

## 5. Ranking and Partitioning

The fifth section will demonstrate how to rank and partition data in Pandas. This is useful for identifying top performers, such as the highest spending customers or the most popular products.

### Pandas

In Pandas, we can use the [`.rank()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html) method to rank values in a DataFrame. This method allows us to specify the ranking method (e.g., dense, average, min, max) and whether to rank in ascending or descending order.

```python {.pandas linenums="1" title="Rank customers by total spending"}
customer_spending: pd.DataFrame = df_sales_pd.groupby("customer_id")["sales_amount"].sum().reset_index()
customer_spending["rank"] = customer_spending["sales_amount"].rank(method="dense", ascending=False)
customer_spending: pd.DataFrame = customer_spending.sort_values("rank")
print(f"Customer Spending Summary: {len(customer_spending)}")
print(customer_spending.head(5))
print(customer_spending.head(5).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="Rank customers by total spending"}
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="Rank customers by total spending"}
customer_spending: psDataFrame = (
    df_sales_ps.groupBy("customer_id")
    .agg(F.sum("sales_amount").alias("total_spending"))
    .withColumn("rank", F.dense_rank().over(Window.orderBy(F.desc("total_spending"))))
    .orderBy("rank")
)
print(f"Customer Spending Summary: {customer_spending.count()}")
customer_spending.show(5)
print(customer_spending.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="Rank customers by total spending"}
customer_spending: pl.DataFrame = (
    df_sales_pl.group_by("customer_id")
    .agg(pl.col("sales_amount").sum().alias("total_spending"))
    .with_columns(pl.col("total_spending").rank(method="dense", descending=True).alias("rank"))
    .sort("rank")
)
print(f"Customer Spending Summary: {len(customer_spending)}")
print(customer_spending.head(5))
print(customer_spending.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

Once we have ranked the customers, we can merge this information with the `customer` DataFrame to get additional details about each customer, such as their name, segment, and city.

### Pandas

```python {.pandas linenums="1" title="TITLE"}
# Add customer details
top_customers: pd.DataFrame = pd.merge(
    customer_spending,
    df_customer_pd[["customer_id", "customer_name", "segment", "city"]],
    on="customer_id",
    how="left",
)
print(f"Top Customers Summary: {len(top_customers)}")
print(top_customers.head(5))
print(top_customers.head(5).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="TITLE"}
# Rank customers by total spending
customer_spending_sql: str = """
    SELECT
        c.customer_id,
        c.customer_name,
        c.segment,
        c.city,
        SUM(s.sales_amount) AS total_spending,
        RANK() OVER (ORDER BY SUM(s.sales_amount) DESC) AS rank
    FROM sales s
    JOIN customer c ON s.customer_id = c.customer_id
    GROUP BY c.customer_id, c.customer_name, c.segment, c.city
    ORDER BY rank
"""
print(f"Customer Spending: {len(pd.read_sql(customer_spending_sql, conn))}")
print(pd.read_sql(customer_spending_sql + "LIMIT 5", conn))
print(pd.read_sql(customer_spending_sql + "LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="TITLE"}
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="TITLE"}
```

<div class="result" markdown>

```txt

```

</div>

Next, we will rank products based on the quantity sold. This allows us to identify the most popular products in terms of sales volume.

### Pandas

```python {.pandas linenums="1" title="Rank products by quantity sold"}
product_popularity: pd.DataFrame = df_sales_pd.groupby("product_id")["quantity"].sum().reset_index()
product_popularity["rank"] = product_popularity["quantity"].rank(method="dense", ascending=False)
product_popularity: pd.DataFrame = product_popularity.sort_values("rank")
print(f"Product Popularity Summary: {len(product_popularity)}")
print(product_popularity.head(5))
print(product_popularity.head(5).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="Rank products by quantity sold"}
product_popularity_sql: str = """
    SELECT
        p.product_id,
        p.product_name,
        p.category,
        SUM(s.quantity) AS total_quantity,
        RANK() OVER (ORDER BY SUM(s.quantity) DESC) AS rank
    FROM sales s
    JOIN product p ON s.product_id = p.product_id
    GROUP BY p.product_id, p.product_name, p.category
    ORDER BY rank
"""
print(f"Product Popularity: {len(pd.read_sql(product_popularity_sql, conn))}")
print(pd.read_sql(product_popularity_sql + "LIMIT 5", conn))
print(pd.read_sql(product_popularity_sql + "LIMIT 5", conn).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="Rank products by quantity sold"}
product_popularity: psDataFrame = (
    df_sales_ps.groupBy("product_id")
    .agg(F.sum("quantity").alias("total_quantity"))
    .withColumn("rank", F.expr("DENSE_RANK() OVER (ORDER BY total_quantity DESC)"))
    .orderBy("rank")
)
print(f"Product Popularity Summary: {product_popularity.count()}")
product_popularity.show(5)
print(product_popularity.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="Rank products by quantity sold"}
product_popularity: pl.DataFrame = (
    df_sales_pl.group_by("product_id")
    .agg(pl.col("quantity").sum().alias("total_quantity"))
    .with_columns(pl.col("total_quantity").rank(method="dense", descending=True).alias("rank"))
    .sort("rank")
)
print(f"Product Popularity Summary: {len(product_popularity)}")
print(product_popularity.head(5))
print(product_popularity.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

As with the customer data, we can merge the product popularity information with the `product` DataFrame to get additional details about each product, such as its name and category.

### Pandas

```python {.pandas linenums="1" title="TITLE"}
# Add product details
top_products: pd.DataFrame = pd.merge(
    product_popularity,
    df_product_pd[["product_id", "product_name", "category"]],
    on="product_id",
    how="left",
)
print(f"Top Products Summary: {len(top_products)}")
print(top_products.head(5))
print(top_products.head(5).to_markdown())
```

<div class="result" markdown>

```txt

```

</div>

### SQL

```python {.sql linenums="1" title="TITLE"}
```

<div class="result" markdown>

```txt

```

</div>

### PySpark

```python {.pyspark linenums="1" title="TITLE"}
```

<div class="result" markdown>

```txt

```

</div>

### Polars

```python {.polars linenums="1" title="TITLE"}
```

<div class="result" markdown>

```txt

```

</div>


## Conclusion

```python
```

<div class="result" markdown>

```txt

```

</div>

[python-print]: https://docs.python.org/3/library/functions.html#print
[pandas]: https://pandas.pydata.org/
[pandas-head]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html
[pandas-read_sql]: https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html
[pandas-subsetting]: https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
[numpy]: https://numpy.org/
[sql-wiki]: https://en.wikipedia.org/wiki/SQL
[sql-iso]: https://www.iso.org/standard/76583.html
[sqlite]: https://sqlite.org/
[sqlite3]: https://docs.python.org/3/library/sqlite3.html
[sqlite3-connect]: https://docs.python.org/3/library/sqlite3.html#sqlite3.connect
[sqlite-where]: https://sqlite.org/lang_select.html#whereclause
[sqlite-select]: https://sqlite.org/lang_select.html
[postgresql]: https://www.postgresql.org/
[mysql]: https://www.mysql.com/
[t-sql]: https://learn.microsoft.com/en-us/sql/t-sql/
[pl-sql]: https://www.oracle.com/au/database/technologies/appdev/plsql.html
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
[hdfs]: https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html
[s3]: https://aws.amazon.com/s3/
[adls]: https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction
[jdbc]: https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html
[polars]: https://www.pola.rs/
[polars-head]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.head.html
[polars-filter]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.filter.html
[polars-filtering]: https://docs.pola.rs/user-guide/transformations/time-series/filter/
[polars-col]: https://docs.pola.rs/api/python/stable/reference/expressions/col.html
[polars-select]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.select.html
